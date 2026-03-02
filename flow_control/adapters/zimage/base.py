import torch
from diffusers import ZImageTransformer2DModel
from einops import rearrange

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
        # ZImageTransformer2DModel expects latents in CBHW instead of BND, we have to
        # do an extra packing and unpacking step here.
        noisy_latents = self._unpack_latents(
            batch["noisy_latents"], h=latent_h, w=latent_w
        )
        # This is uncommon, but it does require CBHW input.
        noisy_latents = rearrange(noisy_latents, "b c h w -> c b h w")
        # Z-Image use 0 for noise, 1 for clean
        timestep = 1 - timestep
        # It requires cap_feats to be a list of tensors without batch dimension
        prompt_embeds = batch["prompt_embeds"].squeeze(0)
        model_pred = self.transformer(
            x=[noisy_latents],
            t=timestep,
            cap_feats=[prompt_embeds],
            return_dict=False,
        )[0]
        model_pred = rearrange(model_pred, "1 c 1 h w -> 1 c h w")
        model_pred = -model_pred  # Negated!
        return self._pack_latents(model_pred)
