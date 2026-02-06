from typing import Any, NotRequired

import torch
from diffusers import LongCatImageTransformer2DModel
from einops import rearrange

from flow_control.adapters.base import BaseModelAdapter
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class BaseLongCatAdapter(BaseModelAdapter):
    """
    Meituan LongCat-Image is a smaller Flux.1-like MMDiT model with Qwen2.5-VL as the text encoder.
    """

    @property
    def transformer(self) -> LongCatImageTransformer2DModel:
        return self.hf_model.model

    @transformer.setter
    def transformer(self, value: Any):
        self.hf_model.model = value

    hf_model: HfModelLoader = HfModelLoader(
        library="diffusers",
        class_name="LongCatImageTransformer2DModel",
        pretrained_model_id="meituan-longcat/LongCat-Image",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    patch_size: int = 2
    vae_scale_factor: int = 8
    image_offset: int = 512

    class BatchType(BaseModelAdapter.BatchType):
        prompt_embeds: torch.Tensor
        """`[B, N, D]` Multimodal embeddings from Qwen2.5-VL-7B."""
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
        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"].shape[1])
        if "img_ids" not in batch:
            scale = self.patch_size * self.vae_scale_factor
            latent_size = (
                batch["image_size"][0] // scale,
                batch["image_size"][1] // scale,
            )
            batch["img_ids"] = self._make_img_ids(
                latent_size,
                index=1,
                h_offset=self.image_offset,
                w_offset=self.image_offset,
            )

        model_pred = self.transformer(
            hidden_states=batch["noisy_latents"],
            timestep=timestep,
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return model_pred

    def _make_txt_ids(self, length: int, index=0, h_offset=0, w_offset=0):
        txt_ids = torch.zeros((length, 3))
        txt_ids[:, 0] = index
        txt_ids[:, 1] = txt_ids[:, 1] + torch.arange(length) + h_offset
        txt_ids[:, 2] = txt_ids[:, 1] + torch.arange(length) + w_offset
        return txt_ids.to(device=self.device)

    def _make_img_ids(
        self, latent_size: tuple[int, int], index=0, h_offset=0, w_offset=0
    ):
        h_len, w_len = latent_size
        img_ids = torch.zeros((h_len, w_len, 3))
        img_ids[:, :, 0] = index
        img_ids[:, :, 1] = (
            img_ids[:, :, 1] + torch.arange(h_len).reshape(h_len, 1) + h_offset
        )
        img_ids[:, :, 2] = (
            img_ids[:, :, 2] + torch.arange(w_len).reshape(1, w_len) + w_offset
        )
        img_ids = rearrange(img_ids, "h w c -> (h w) c")
        return img_ids.to(device=self.device)
