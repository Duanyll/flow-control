import torch
from diffusers import ZImageTransformer2DModel

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class ZImageBatch(Batch):
    prompt_embeds: torch.Tensor
    """`[B, N, D]` Text embeddings from the Qwen3VL text encoder."""


class ZImageAdapter[TBatch: ZImageBatch](
    BaseModelAdapter[ZImageTransformer2DModel, TBatch]
):
    vae_scale_factor: int = 8
    hf_model: HfModelLoader[ZImageTransformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="ZImageTransformer2DModel",
        pretrained_model_id="Tongyi-MAI/Z-Image",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    def predict_velocity(self, batch, timestep):
        latent_h, latent_w = (
            batch["image_size"][0] // self.vae_scale_factor,
            batch["image_size"][1] // self.vae_scale_factor,
        )
        # ZImageTransformer2DModel expects latents in BCHW instead of BND, we have to
        # do an extra packing and unpacking step here.
        noisy_latents = self._unpack_latents(
            batch["noisy_latents"], h=latent_h, w=latent_w
        )
        # Z-Image use 0 for noise, 1 for clean
        timestep = 1 - timestep
        prompt_embeds = batch["prompt_embeds"]
        model_pred = self.transformer(
            x=[noisy_latents],
            t=timestep,
            cap_feats=[prompt_embeds],
            return_dict=False,
        )[0]
        return self._pack_latents(model_pred)
