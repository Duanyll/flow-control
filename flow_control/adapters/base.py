"""Model adapters: what one DiT family implements, and how its transformer is loaded."""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Protocol, TypedDict

import torch
from diffusers import ModelMixin
from einops import rearrange
from pydantic import ConfigDict
from transformers import PreTrainedModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.registry import Registry
from flow_control.utils.types import TorchDType
from flow_control.utils.upcasting import (
    apply_layerwise_upcasting,
    cast_trainable_parameters,
)

from .batching import MicrobatchMixin
from .weights import VariantsMixin

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

    def use_variant(self, variant: str | None, /) -> AbstractContextManager[None]: ...

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
    VariantsMixin, MicrobatchMixin, ABC
):
    """One model family: its transformer, latent layout and velocity forward.

    `VariantsMixin` adds the weight variants, `MicrobatchMixin` the batched
    forward every consumer calls.
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------ configuration ------------------------------ #

    arch: str
    type: str
    hf_model: HfModelLoader[TModel]
    storage_dtype: TorchDType | None = None
    """Specify a storage dtype (e.g. float8_e4m3fn) to apply layerwise upcasting. """
    trainable_dtype: TorchDType = torch.bfloat16
    """The dtype to cast trainable parameters to."""
    # TODO: Add standard PyTorch AMP (torch.autocast) support (bf16 activation + fp32 trainable params)

    patch_size: int = 2
    vae_scale_factor: int = 8
    latent_channels: int = 16

    # ---------------------------- the loaded model ----------------------------- #

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

    @property
    def dtype(self) -> torch.dtype:
        # Ensure we are getting the correct dtype even after upcasting
        return (
            self.hf_model.dtype
            if self.hf_model.dtype != "auto"
            else self.transformer.dtype
        )

    def load_transformer(self, device: torch.device) -> None:
        """Load the weights, add the family's modules, realize the variants, set dtypes."""
        if not self.hf_model.load_model(device=device, frozen=not self.trainable):
            self._attach()  # another adapter already realized the cached transformer
            return
        self._install_modules()
        self._realize(device)
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

    # -------------------------- what a family implements ----------------------- #

    def _install_modules(self):
        """Create extra modules on the freshly loaded base model, before the variants."""
        pass

    @abstractmethod
    def _predict_velocity(
        self,
        batch: TBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """One forward through `self.module()`, for one or more collated logical samples."""
        raise NotImplementedError()

    def cost_test(self):
        """Yield synthetic batches of increasing ``cost`` (token total) for
        ``SftTrainer.run_cost_test``'s memory probe."""
        raise NotImplementedError()

    # ------------------------------ latent layout ------------------------------ #

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


adapter_registry: Registry[BaseModelAdapter] = Registry(
    "model_adapter", base=BaseModelAdapter
)
