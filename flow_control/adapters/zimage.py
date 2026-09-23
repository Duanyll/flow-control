from typing import Literal

import torch
from diffusers import ZImageTransformer2DModel
from einops import rearrange

from flow_control.adapters.base import BaseModelAdapter, Batch, adapter_registry
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class ZImageBatch(Batch):
    prompt_embeds: torch.Tensor
    """`[B, N, D]` Text embeddings from the Qwen3VL text encoder."""


@adapter_registry.register("zimage_base")
class ZImageAdapter[TBatch: ZImageBatch](
    BaseModelAdapter[ZImageTransformer2DModel, TBatch]
):
    supports_dense_batching = True
    dense_batch_fields = (
        "image_size",
        "noisy_latents",
        "prompt_embeds",
    )
    arch: Literal["zimage"] = "zimage"
    type: Literal["base"] = "base"

    vae_scale_factor: int = 8
    hf_model: HfModelLoader[ZImageTransformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="ZImageTransformer2DModel",
        pretrained_model_id="Tongyi-MAI/Z-Image",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    def _predict_velocity(self, batch, timestep):
        latent_h, latent_w = (
            batch["image_size"][0] // self.vae_scale_factor,
            batch["image_size"][1] // self.vae_scale_factor,
        )
        # ZImageTransformer2DModel expects latents in CBHW instead of BND, we have to
        # do an extra packing and unpacking step here.
        noisy_latents = self._unpack_latents(
            batch["noisy_latents"], h=latent_h, w=latent_w
        )
        # The native API accepts one CBHW tensor per logical sample.
        latent_inputs = [
            rearrange(latent, "1 c h w -> c 1 h w")
            for latent in noisy_latents.split(1, dim=0)
        ]
        # Z-Image use 0 for noise, 1 for clean
        timestep = 1 - timestep
        prompt_embeds = list(batch["prompt_embeds"].unbind(dim=0))
        model_preds = self.transformer(
            x=latent_inputs,
            t=timestep,
            cap_feats=prompt_embeds,
            return_dict=False,
        )[0]
        packed_predictions = [
            self._pack_latents(-rearrange(prediction, "c 1 h w -> 1 c h w"))
            for prediction in model_preds
        ]
        return torch.cat(packed_predictions, dim=0)
