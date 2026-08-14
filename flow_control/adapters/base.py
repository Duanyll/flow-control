from abc import ABC, abstractmethod
from typing import ClassVar, TypedDict, cast

import torch
import torch.distributed as dist
from diffusers import ModelMixin
from einops import rearrange
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.registry import Registry
from flow_control.utils.tensor import deep_cast_float_dtype, deep_move_to_device
from flow_control.utils.types import TorchDType
from flow_control.utils.upcasting import (
    apply_layerwise_upcasting,
    cast_trainable_parameters,
)

logger = get_logger(__name__)


class Batch(TypedDict):
    image_size: tuple[int, int]
    """`(H, W)` The size of the image to generate."""
    clean_latents: torch.Tensor
    """`[B, N, D]` The clean latents. Only available during training."""
    noisy_latents: torch.Tensor
    """`[B, N, D]` The noisy latents to denoise."""


class BaseModelAdapter[TModel: ModelMixin | PreTrainedModel, TBatch: Batch](
    BaseModel, ABC
):
    """
    Base class for all control adapters.
    """

    arch: str
    type: str

    model_config = ConfigDict(extra="forbid")

    @property
    def transformer(self) -> TModel:
        return self.hf_model.model

    @transformer.setter
    def transformer(self, value: TModel) -> None:
        self.hf_model.model = value

    @property
    def device(self) -> torch.device:
        return self.transformer.device

    hf_model: HfModelLoader[TModel]
    storage_dtype: TorchDType | None = None
    """Specify a storage dtype (e.g. float8_e4m3fn) to apply layerwise upcasting. """
    trainable_dtype: TorchDType = torch.bfloat16
    """The dtype to cast trainable parameters to."""
    # TODO: Add standard PyTorch AMP (torch.autocast) support (bf16 activation + fp32 trainable params)

    all_trainable: bool = False
    peft_lora_config: LoraConfig = LoraConfig()
    peft_lora_rank: int = 0
    """If > 0, will apply PEFT LoRA adapters with the given rank. Overrides `r` in `peft_lora_config`."""
    extra_trainable_modules: list[str] = []
    """
    List of module name substrings to make trainable, in addition to any PEFT adapters. 
    Matches if the substring is contained in the parameter's FQN.
    """

    patch_size: int = 2
    latent_channels: int = 16

    supports_dense_batching: ClassVar[bool] = False
    """Whether equal-shaped logical samples may use the default dense collator."""
    dense_batch_fields: ClassVar[tuple[str, ...]] = ()
    """Adapter inputs used by the default collator; unrelated metadata is ignored."""

    @property
    def dtype(self) -> torch.dtype:
        # Ensure we are getting the correct dtype even after upcasting
        return (
            self.hf_model.dtype
            if self.hf_model.dtype != "auto"
            else self.transformer.dtype
        )

    def load_transformer(self, device: torch.device) -> None:
        freshly_loaded = self.hf_model.load_model(
            device=device, frozen=not self.all_trainable
        )
        if not freshly_loaded:
            return  # reused from cache, post-load already done

        self._install_modules()

        if self.peft_lora_rank > 0:
            self.peft_lora_config.r = self.peft_lora_rank
            if self.peft_lora_config.target_modules == "all-linear":
                self.peft_lora_config.target_modules = list(
                    {
                        k
                        for k, v in self.transformer.named_modules()
                        if isinstance(v, torch.nn.Linear)
                    }
                )
            self.transformer.add_adapter(self.peft_lora_config)

        for name, param in self.transformer.named_parameters():
            if any(k in name for k in self.extra_trainable_modules):
                param.requires_grad = True

        cast_trainable_parameters(self.transformer, self.trainable_dtype)
        if (
            self.hf_model.dtype != "auto"
            and self.storage_dtype is not None
            and self.storage_dtype != self.hf_model.dtype
        ):
            apply_layerwise_upcasting(
                self.transformer,
                storage_dtype=self.storage_dtype,
                compute_dtype=self.hf_model.dtype,
            )
            logger.info(
                f"Applied layerwise casting with storage dtype {self.storage_dtype} and compute dtype {self.hf_model.dtype}"
            )

    def _install_modules(self):
        """
        Create and initialize additional modules on the base model. Called after base model is
        created, before installing PEFT adapters and upcasting.
        """
        pass

    @abstractmethod
    def _predict_velocity(
        self,
        batch: TBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _prepare_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _same_static_value(left: object, right: object) -> bool:
        if type(left) is not type(right):
            return False
        try:
            equal = left == right
        except (RuntimeError, TypeError, ValueError):
            return False
        return bool(equal) if isinstance(equal, bool) else False

    def _collate_velocity_inputs(
        self,
        batches: list[TBatch],
        timesteps: list[torch.Tensor],
    ) -> tuple[TBatch, torch.Tensor] | None:
        """Densely collate fixed-shape batches, or return ``None`` to fall back."""
        if not self.supports_dense_batching:
            return None

        collated: dict[str, object] = {}
        for key in self.dense_batch_fields:
            present = [key in batch for batch in batches]
            if not any(present):
                continue
            if not all(present):
                return None
            values = [batch[key] for batch in batches]
            success, value = self._collate_velocity_values(values)
            if not success:
                return None
            collated[key] = value

        return cast(TBatch, collated), torch.cat(timesteps, dim=0)

    def _collate_velocity_values(
        self,
        values: list[object],
    ) -> tuple[bool, object]:
        first = values[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                return False, first
            tensors = cast(list[torch.Tensor], values)
            if any(
                tensor.ndim == 0 or tensor.shape[0] != 1 or tensor.shape != first.shape
                for tensor in tensors
            ):
                return False, first
            return True, torch.cat(tensors, dim=0)

        if isinstance(first, list):
            if not all(
                isinstance(value, list) and len(value) == len(first) for value in values
            ):
                return False, first
            collated_items: list[object] = []
            lists = cast(list[list[object]], values)
            for index in range(len(first)):
                success, item = self._collate_velocity_values(
                    [value[index] for value in lists]
                )
                if not success:
                    return False, first
                collated_items.append(item)
            return True, collated_items

        if all(self._same_static_value(first, value) for value in values[1:]):
            return True, first
        return False, first

    def _sync_collation_decision(
        self,
        can_collate: bool,
        logical_batch_size: int,
    ) -> bool:
        if not dist.is_initialized():
            return can_collate
        # One MIN reduction communicates the decision, min length, and max length.
        status = torch.tensor(
            [int(can_collate), logical_batch_size, -logical_batch_size],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if int(status[1].item()) != -int(status[2].item()):
            raise ValueError(
                "All distributed ranks must submit the same number of logical "
                "samples to predict_velocity_batched."
            )
        return bool(status[0].item())

    def predict_velocity_batched(
        self,
        batches: list[TBatch],
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Predict one velocity per logical sample.

        Logical samples retain their leading singleton batch dimension. Adapters that
        opt into dense batching combine compatible samples into one physical forward;
        all other inputs use the synchronized sample-at-a-time fallback.
        """
        if not batches:
            raise ValueError("predict_velocity_batched requires at least one batch.")
        if len(batches) != len(timesteps):
            raise ValueError(
                "batches and timesteps must have equal lengths, got "
                f"{len(batches)} and {len(timesteps)}."
            )
        for index, batch in enumerate(batches):
            if batch["noisy_latents"].shape[0] != 1:
                raise ValueError(
                    "Each logical sample must have a singleton leading batch "
                    f"dimension; sample {index} has shape "
                    f"{tuple(batch['noisy_latents'].shape)}."
                )

        prepared_batches = [
            cast(
                TBatch,
                deep_move_to_device(
                    deep_cast_float_dtype(batch, self.dtype), self.device
                ),
            )
            for batch in batches
        ]
        prepared_timesteps = [
            self._prepare_timestep(timestep) for timestep in timesteps
        ]
        collated = self._collate_velocity_inputs(prepared_batches, prepared_timesteps)

        if self._sync_collation_decision(collated is not None, len(batches)):
            assert collated is not None
            collated_batch, collated_timestep = collated
            velocity = self._predict_velocity(collated_batch, collated_timestep).float()
            if velocity.ndim == 0 or velocity.shape[0] != len(batches):
                raise ValueError(
                    "Batched adapter output has the wrong leading dimension: "
                    f"expected {len(batches)}, got {tuple(velocity.shape)}."
                )
            return list(velocity.split(1, dim=0))

        velocities = [
            self._predict_velocity(batch, timestep).float()
            for batch, timestep in zip(
                prepared_batches, prepared_timesteps, strict=True
            )
        ]
        for index, velocity in enumerate(velocities):
            if velocity.ndim == 0 or velocity.shape[0] != 1:
                raise ValueError(
                    "Fallback adapter output must retain a singleton leading "
                    f"dimension; sample {index} has shape {tuple(velocity.shape)}."
                )
        return velocities

    def _pack_latents(self, latents):
        return rearrange(
            latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def _unpack_latents(self, latents, h, w):
        return rearrange(
            latents,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h // self.patch_size,
            w=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def latent_length_test(self):
        raise NotImplementedError()


adapter_registry: Registry[BaseModelAdapter] = Registry(
    "model_adapter", base=BaseModelAdapter
)
