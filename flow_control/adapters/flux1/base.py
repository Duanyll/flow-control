from typing import Any, NotRequired

import torch
from diffusers import FluxTransformer2DModel
from einops import rearrange

from flow_control.adapters.base import BaseModelAdapter
from flow_control.utils.loaders import HfModelLoader
from flow_control.utils.logging import get_logger

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

    guidance: float = 3.5
    patch_size: int = 2
    vae_scale_factor: int = 8

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

    def predict_velocity(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = torch.full((b,), self.guidance, device=device)

        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            batch["img_ids"] = self._make_img_ids(batch["image_size"])

        model_pred = self.transformer(
            hidden_states=batch["noisy_latents"],
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return model_pred

    def _make_txt_ids(self, prompt_embeds):
        b, n, d = prompt_embeds.shape
        return torch.zeros(
            (n, 3), dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )

    def _make_img_ids(
        self, image_size: tuple[int, int], index=0, h_offset=0, w_offset=0
    ):
        h, w = image_size
        h_len = h // (self.patch_size * self.vae_scale_factor)
        w_len = w // (self.patch_size * self.vae_scale_factor)
        img_ids = torch.zeros((h_len, w_len, 3), dtype=self.dtype, device=self.device)
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
