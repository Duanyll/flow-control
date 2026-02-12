from typing import NotRequired

import torch
from diffusers import FluxTransformer2DModel
from einops import rearrange
from peft import LoraConfig

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class Flux1Batch(Batch):
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


class Flux1Adapter[TBatch: Flux1Batch](
    BaseModelAdapter[FluxTransformer2DModel, TBatch]
):
    hf_model: HfModelLoader[FluxTransformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="FluxTransformer2DModel",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    guidance: float | None = 3.5
    """
    Guidance scale for DISTILLED classifier-free guidance as timestep embeddings.
    """
    patch_size: int = 2
    vae_scale_factor: int = 8
    peft_lora_config: LoraConfig = LoraConfig(
        target_modules=[
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    )

    def predict_velocity(
        self,
        batch: TBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = (
            torch.full((b,), self.guidance, device=device)
            if self.guidance is not None
            else None
        )

        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            scale = self.patch_size * self.vae_scale_factor
            latent_size = (
                batch["image_size"][0] // scale,
                batch["image_size"][1] // scale,
            )
            batch["img_ids"] = self._make_img_ids(latent_size)

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

    def _make_txt_ids(self, prompt_embeds: torch.Tensor):
        b, n, d = prompt_embeds.shape
        return torch.zeros(
            (n, 3), dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )

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
        return img_ids.to(dtype=self.dtype, device=self.device)
