from typing import Any, NotRequired

import torch
from diffusers import FluxTransformer2DModel
from einops import rearrange, reduce

from flow_control.adapters.base import BaseModelAdapter
from flow_control.utils.loaders import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.types import TorchDType
from flow_control.utils.upcasting import (
    apply_layerwise_upcasting,
    cast_trainable_parameters,
)

logger = get_logger(__name__)


class BaseFlux1Adapter(BaseModelAdapter):
    @property
    def transformer(self) -> FluxTransformer2DModel:
        return self._transformer  # type: ignore

    @transformer.setter
    def transformer(self, value: Any):
        self._transformer = value

    hf_model: HfModelLoader = HfModelLoader(
        type="diffusers",
        class_name="FluxTransformer2DModel",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )
    storage_dtype: TorchDType | None = None
    trainable_dtype: TorchDType = torch.bfloat16

    all_trainable: bool = False

    guidance: float = 3.5
    patch_size: int = 2

    default_latent_resolution: tuple[int, int] = (128, 128)
    latent_channels: int = 16

    class BatchType(BaseModelAdapter.BatchType):
        pooled_prompt_embeds: torch.Tensor
        """`[B, D]` Pooled text embeddings from the CLIP text encoder."""
        prompt_embeds: torch.Tensor
        """`[B, N, D]` Text embeddings from T5XXL text encoder."""
        txt_ids: NotRequired[torch.Tensor]
        """`[B, N, 3]` Used for adding positional embeddings to the text embeddings.
           Usually all zeros. Will be calculated if not present."""
        img_ids: NotRequired[torch.Tensor]
        """`[B, N, 3]` Used for adding positional embeddings to the image embeddings.
           Will be calculated if not present."""

    @property
    def dtype(self) -> torch.dtype:
        # Ensure we are getting the correct dtype even after upcasting
        return (
            self.hf_model.dtype
            if self.hf_model.dtype != "auto"
            else self.transformer.dtype
        )

    def load_transformer(self):
        self.transformer = self.hf_model.load_model()  # type: ignore
        self.transformer.requires_grad_(self.all_trainable)
        self._install_modules()
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

    def predict_velocity(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = torch.full((b,), self.guidance, device=device)

        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])

        model_pred = self.transformer(
            hidden_states=self._pack_latents(batch["noisy_latents"]),
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return self._unpack_latents(model_pred, h, w)

    def generate_noise(
        self,
        batch: BatchType,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if "clean_latents" in batch:
            clean = batch["clean_latents"]
            if generator is None:
                return torch.empty_like(clean).normal_(generator=generator)
            else:
                return torch.randn_like(clean)
        else:
            noise = torch.randn(
                (1, self.latent_channels, *self.default_latent_resolution),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            return noise

    def get_latent_length(self, batch: BatchType) -> int:
        b, c, h, w = batch["noisy_latents"].shape
        latent_len = (h // self.patch_size) * (w // self.patch_size)
        return latent_len

    def train_step(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        clean = batch["clean_latents"]
        if "noisy_latents" not in batch:
            noise = self.generate_noise(batch)
            batch["noisy_latents"] = (1.0 - timestep) * clean + timestep * noise
        noise = batch["noisy_latents"]

        model_pred = self.predict_velocity(batch, timestep)
        target = noise - clean
        loss = reduce(
            (model_pred.float() - target.float()) ** 2, "b c h w -> b", reduction="mean"
        )  # Must use float() here
        return loss

    def _make_txt_ids(self, prompt_embeds):
        b, n, d = prompt_embeds.shape
        return torch.zeros(
            (n, 3), dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )

    def _make_img_ids(self, pixel_latents, index=0, h_offset=0, w_offset=0):
        b, c, h, w = pixel_latents.shape
        h_len = h // self.patch_size
        w_len = w // self.patch_size
        img_ids = torch.zeros(
            (h_len, w_len, 3), dtype=pixel_latents.dtype, device=pixel_latents.device
        )
        img_ids[:, :, 0] = index
        img_ids[:, :, 1] = (
            img_ids[:, :, 1]
            + torch.arange(h_len, dtype=img_ids.dtype, device=img_ids.device).reshape(
                h_len, 1
            )
            + h_offset // self.patch_size
        )
        img_ids[:, :, 2] = (
            img_ids[:, :, 2]
            + torch.arange(w_len, dtype=img_ids.dtype, device=img_ids.device).reshape(
                1, w_len
            )
            + w_offset // self.patch_size
        )
        img_ids = rearrange(img_ids, "h w c -> (h w) c")
        return img_ids

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
