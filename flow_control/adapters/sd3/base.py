from typing import Literal

import torch
from diffusers import SD3Transformer2DModel
from peft import LoraConfig

from flow_control.adapters.base import BaseModelAdapter, Batch, adapter_registry
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class SD3Batch(Batch):
    prompt_embeds: torch.Tensor
    """`[B, N, 4096]` Fused CLIP-L / CLIP-G / T5 sequence conditioning."""
    pooled_prompt_embeds: torch.Tensor
    """`[B, 2048]` Fused CLIP-L / CLIP-G pooled projections."""


@adapter_registry.register("sd3_base")
class SD3Adapter[TBatch: SD3Batch](BaseModelAdapter[SD3Transformer2DModel, TBatch]):
    """Adapter for Stable Diffusion 3 / 3.5 (``SD3Transformer2DModel``).

    Two differences from the FLUX adapter:

    - SD3's transformer consumes **raw ``BCHW`` latents** (it patchifies internally),
      whereas the framework carries pre-packed ``[B, N, D]`` latents. So we unpack
      before the forward pass and re-pack the predicted velocity.
    - SD3 is **not** guidance-distilled: there is no ``guidance`` embedding and true CFG
      is applied by the sampler (``cfg_scale > 1`` with a negative batch).
    """

    arch: Literal["sd3"] = "sd3"
    type: Literal["base"] = "base"
    hf_model: HfModelLoader[SD3Transformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="SD3Transformer2DModel",
        pretrained_model_id="stabilityai/stable-diffusion-3.5-medium",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    patch_size: int = 2
    latent_channels: int = 16
    vae_scale_factor: int = 8
    peft_lora_config: LoraConfig = LoraConfig(
        target_modules=[
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
        ]
    )

    def _predict_velocity(
        self,
        batch: TBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        h, w = batch["image_size"]
        latents = self._unpack_latents(
            batch["noisy_latents"],
            h // self.vae_scale_factor,
            w // self.vae_scale_factor,
        )

        # The sampler passes ``timestep`` as sigma in [0, 1]. FLUX's transformer rescales
        # to [0, 1000] internally; SD3's does not, so we scale it here to match the range
        # SD3 was trained with.
        model_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep * 1000,
            encoder_hidden_states=batch["prompt_embeds"],
            pooled_projections=batch["pooled_prompt_embeds"],
            return_dict=False,
        )[0]

        return self._pack_latents(model_pred)
