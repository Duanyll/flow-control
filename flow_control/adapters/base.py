import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, ClassVar, Protocol, TypedDict, cast

import torch
import torch.distributed as dist
from diffusers import ModelMixin
from einops import rearrange
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from pydantic import BaseModel, ConfigDict, PositiveInt, PrivateAttr
from transformers import PreTrainedModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.model_cache import cache_enabled, cache_fields, clear_cache
from flow_control.utils.registry import Registry
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_detach,
    deep_move_to_device,
)
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


class SamplerModel(Protocol):
    """Minimal model interface consumed by sampling and replay."""

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def micro_batch_size(self) -> int: ...

    def use_variant(self, variant: str | None) -> AbstractContextManager[None]: ...

    def predict_velocity_batched(
        self,
        batches: list[Any],
        timesteps: list[torch.Tensor],
        *,
        dummy_outputs: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Return fp32 velocities; lower precision computation stays inside the adapter."""
        ...


class BaseModelAdapter[TModel: ModelMixin | PreTrainedModel, TBatch: Batch](
    BaseModel, ABC
):
    """
    Base class for all control adapters.
    """

    arch: str
    type: str

    model_config = ConfigDict(extra="forbid")
    _base_variant_depth: int = PrivateAttr(default=0)
    _active_variant: str | None = PrivateAttr(default=None)
    _dummy_sample: tuple[TBatch, torch.Tensor] | None = PrivateAttr(default=None)
    """Last real prepared sample, forwarded in place of samples another rank
    has and this rank lacks. Pins that sample's tensors on the device until
    the next call; only set in multi-rank runs, where a dummy can be needed."""

    @contextmanager
    def use_variant(self, variant: str | None) -> Iterator[None]:
        """Select loaded LoRA weights, restoring selection and trainability.

        A surrounding base/reference context dominates branch-level choices.
        PEFT's public toggles change requires_grad, so restore those flags
        before forward too: FSDP's parameter membership must remain fixed.
        """
        if (
            variant is None
            or self._base_variant_depth
            or variant == self._active_variant
        ):
            yield
            return
        transformer = self.transformer
        if variant != "base" and variant not in getattr(transformer, "peft_config", {}):
            raise ValueError(f"Model variant {variant!r} is not a loaded LoRA adapter.")
        layers = [
            module
            for module in transformer.modules()
            if isinstance(module, BaseTunerLayer)
        ]
        selections = [
            (layer, list(layer.active_adapters), layer.disable_adapters)
            for layer in layers
        ]
        trainability = [
            (parameter, parameter.requires_grad)
            for parameter in transformer.parameters()
        ]
        previous_variant = self._active_variant
        self._active_variant = variant
        if variant == "base":
            self._base_variant_depth += 1
        try:
            for layer in layers:
                if variant != "base":
                    layer.set_adapter(variant)
                layer.enable_adapters(variant != "base")
            for parameter, requires_grad in trainability:
                parameter.requires_grad_(requires_grad)
            with self._checkpoint_variant(variant):
                yield
        finally:
            for layer, names, disabled in selections:
                layer.set_adapter(names)
                layer.enable_adapters(not disabled)
            for parameter, requires_grad in trainability:
                parameter.requires_grad_(requires_grad)
            if variant == "base":
                self._base_variant_depth -= 1
            self._active_variant = previous_variant

    @contextmanager
    def _checkpoint_variant(self, variant: str) -> Iterator[None]:
        checkpoints = [
            (module, checkpoint)
            for module in self.transformer.modules()
            if callable(
                checkpoint := getattr(module, "_gradient_checkpointing_func", None)
            )
        ]

        def capture(checkpoint: Callable[..., Any]) -> Callable[..., Any]:
            def checkpoint_with_variant(
                function: Callable[..., Any], *args: Any, **kwargs: Any
            ) -> Any:
                def call_with_variant(*inputs: Any, **call_kwargs: Any) -> Any:
                    # Checkpoint recomputation runs after the selecting context exits.
                    with self.use_variant(variant):
                        return function(*inputs, **call_kwargs)

                return checkpoint(call_with_variant, *args, **kwargs)

            return checkpoint_with_variant

        try:
            for module, checkpoint in checkpoints:
                cast(Any, module)._gradient_checkpointing_func = capture(checkpoint)
            yield
        finally:
            for module, checkpoint in checkpoints:
                cast(Any, module)._gradient_checkpointing_func = checkpoint

    @property
    def transformer(self) -> TModel:
        return self.hf_model.model

    @transformer.setter
    def transformer(self, value: TModel) -> None:
        self.hf_model.model = value

    @property
    def device(self) -> torch.device:
        # Both bounds define `.device`, but ty checks the property descriptor
        # against the whole `ModelMixin | PreTrainedModel` receiver at once.
        return self.transformer.device  # ty: ignore[invalid-attribute-access]

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
    vae_scale_factor: int = 8
    latent_channels: int = 16

    micro_batch_size: PositiveInt = 1
    """Maximum logical samples per chunk. A throughput/memory knob: chunking is
    mathematically equivalent, but dense forwards are not bitwise reproducible
    on GPU."""

    use_cache: bool = True
    """Reuse adapter intermediates within a no-grad Executor run."""

    shared_cache_fields: ClassVar[tuple[str, ...]] = ()
    """Layout-only caches shared by compatible dense samples, independent of B.
    Sample-dependent caches require adapter-specific collation and splitting."""

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
            # Configuring LoRA at all requires a PeftAdapterMixin transformer,
            # which neither bound guarantees.
            self.transformer.add_adapter(  # ty: ignore[invalid-argument-type]
                self.peft_lora_config
            )

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

        for key in self.shared_cache_fields:
            for batch in batches:
                if key in batch:
                    collated[key] = batch[key]
                    break
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

    def _sync_max(self, values: list[int]) -> list[int]:
        """MAX-reduce small integers across ranks."""
        if not dist.is_initialized():
            return values
        status = torch.tensor(values, device=self.device, dtype=torch.int64)
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        return status.tolist()

    def _checked_velocity(self, velocity: torch.Tensor, expected: int) -> torch.Tensor:
        if velocity.ndim == 0 or velocity.shape[0] != expected:
            raise ValueError(
                f"{type(self).__name__}._predict_velocity returned shape "
                f"{tuple(velocity.shape)}, expected leading dimension {expected}."
            )
        return velocity.float()

    def _forward_one(self, batch: TBatch, timestep: torch.Tensor) -> torch.Tensor:
        return self._checked_velocity(self._predict_velocity(batch, timestep), 1)

    def _forward_dummy(self) -> torch.Tensor:
        assert self._dummy_sample is not None  # Seeded collectively before chunking.
        try:
            return self._forward_one(*self._dummy_sample)
        finally:
            clear_cache(self._dummy_sample[0])

    def _share_dummy(self, source: int) -> None:
        """Seed empty ranks once, transferring detached CPU data rather than GPU objects."""
        payload: list[Any] = [
            deep_move_to_device(self._dummy_sample, torch.device("cpu"))
            if dist.get_rank() == source
            else None
        ]
        dist.broadcast_object_list(payload, src=source, device=self.device)
        if self._dummy_sample is None:
            self._dummy_sample = deep_move_to_device(payload[0], self.device)

    def _forward_chunk(
        self,
        batches: list[TBatch],
        timesteps: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run one chunk as ``(velocities, dummy_outputs)``.

        A chunk is one dense forward when every rank holding data can collate
        it, else as many sequential forwards as the longest rank's chunk. This
        rank pads its shortfall (an empty chunk, or fewer samples than the
        longest rank) with forwards on the dummy sample.
        """
        collated = (
            self._collate_velocity_inputs(batches, timesteps) if batches else None
        )
        needs_fallback = bool(batches) and collated is None
        sequential, longest = self._sync_max([int(needs_fallback), len(batches)])
        if not sequential:
            if collated is None:  # Empty chunk: pad the peers' single dense forward.
                return [], [self._forward_dummy()]
            try:
                velocity = self._checked_velocity(
                    self._predict_velocity(*collated), len(batches)
                )
                shared = {
                    key: collated[0][key]
                    for key in self.shared_cache_fields
                    if key in collated[0]
                }
                for batch in batches:
                    cast(dict[str, Any], batch).update(shared)
                return list(velocity.split(1, dim=0)), []
            finally:
                clear_cache(collated[0])
        velocities = [
            self._forward_one(batch, timestep)
            for batch, timestep in zip(batches, timesteps, strict=True)
        ]
        dummies = [self._forward_dummy() for _ in range(longest - len(batches))]
        return velocities, dummies

    @contextmanager
    def _prepared_batches(self, batches: list[TBatch]) -> Iterator[list[TBatch]]:
        """Prepare model inputs; write runtime fields back only inside Executor."""
        reuse_cache = (
            self.use_cache and cache_enabled.get() and not torch.is_grad_enabled()
        )
        prepared_batches: list[TBatch] = []
        for batch in batches:
            data = {
                key: value for key, value in batch.items() if not key.startswith("_")
            }
            prepared = deep_move_to_device(
                deep_cast_float_dtype(data, self.dtype), self.device
            )
            if reuse_cache:
                # Cached values already have the model's device/dtype. Keep
                # their identity, including structured K/V, across preparation.
                prepared.update(cache_fields(batch))
            prepared_batches.append(cast(TBatch, prepared))
        try:
            yield prepared_batches
            if reuse_cache:
                for batch, prepared in zip(batches, prepared_batches, strict=True):
                    cast(dict[str, Any], batch).update(cache_fields(prepared))
        finally:
            for prepared in prepared_batches:
                clear_cache(prepared)

    def predict_velocity_batched(
        self,
        batches: list[TBatch],
        timesteps: list[torch.Tensor],
        *,
        dummy_outputs: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Predict one velocity per logical sample; a collective on every rank.

        Logical samples keep their singleton leading batch dimension. Every
        rank runs the same chunk count and, within a chunk, the same forward
        count (see ``_forward_chunk``), padding with dummy forwards. An empty
        list is therefore a legal call returning ``[]``. Under autograd, dummy
        outputs are folded into the first real output with zero weight so FSDP
        backward stays aligned across ranks. A caller grouping multiple variants
        can supply ``dummy_outputs`` and fold them into its combined result;
        this also supports variants with no local real output.
        """
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

        with self._prepared_batches(batches) as prepared_batches:
            prepared_timesteps = [
                self._prepare_timestep(timestep) for timestep in timesteps
            ]
            if prepared_batches and dist.is_initialized() and dist.get_world_size() > 1:
                self._dummy_sample = (
                    deep_detach(
                        {
                            key: value
                            for key, value in prepared_batches[0].items()
                            if not key.startswith("_")
                        }
                    ),
                    prepared_timesteps[0].detach(),
                )
            size = self.micro_batch_size
            chunks, needs_dummy, donor = self._sync_max(
                [
                    math.ceil(len(prepared_batches) / size),
                    int(self._dummy_sample is None),
                    dist.get_rank() + 1
                    if dist.is_initialized() and self._dummy_sample is not None
                    else 0,
                ]
            )
            if chunks and needs_dummy and donor:
                self._share_dummy(donor - 1)
            velocities: list[torch.Tensor] = []
            dummies: list[torch.Tensor] = []
            # Run the longest rank's chunk count; slicing past our own end is empty.
            for start in range(0, chunks * size, size):
                chunk_velocities, chunk_dummies = self._forward_chunk(
                    prepared_batches[start : start + size],
                    prepared_timesteps[start : start + size],
                )
                velocities.extend(chunk_velocities)
                dummies.extend(chunk_dummies)
            dummies = [output for output in dummies if output.requires_grad]
            if dummy_outputs is not None:
                dummy_outputs.extend(dummies)
            elif dummies:
                if not velocities:
                    raise RuntimeError(
                        f"Rank {dist.get_rank()} ran dummy forwards under autograd "
                        "with no real sample to carry their zero-weight graph "
                        "dependency; a training call must give every rank at least "
                        "one item."
                    )
                velocities[0] = velocities[0] + sum(
                    output.sum() * 0 for output in dummies
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

    def cost_test(self):
        """Yield synthetic batches of increasing ``cost`` (token total) for
        ``SftTrainer.run_cost_test``'s memory probe."""
        raise NotImplementedError()


adapter_registry: Registry[BaseModelAdapter] = Registry(
    "model_adapter", base=BaseModelAdapter
)
