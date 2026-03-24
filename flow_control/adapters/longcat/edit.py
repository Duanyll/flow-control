from typing import Literal

import torch
from diffusers import LongCatImageTransformer2DModel

from flow_control.utils.hf_model import HfModelLoader

from .base import LongCatAdapter, LongCatBatch


class LongCatEditBatch(LongCatBatch):
    reference_latents: list[torch.Tensor]
    """List of `[B, N, D]` Tensors representing VAE encoded reference images."""
    reference_sizes: list[tuple[int, int]]
    """List of `(H, W)` tuples representing the sizes of the reference images."""


class LongCatEditAdapter(LongCatAdapter[LongCatEditBatch]):
    type: Literal["edit"] = "edit"
    hf_model: HfModelLoader[LongCatImageTransformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="LongCatImageTransformer2DModel",
        pretrained_model_id="meituan-longcat/LongCat-Image-Edit",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    def _predict_velocity(
        self, batch: LongCatEditBatch, timestep: torch.Tensor
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape

        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"].shape[1])
        if "img_ids" not in batch:
            prompt_len = batch["prompt_embeds"].shape[1]
            scale = self.patch_size * self.vae_scale_factor
            latent_size = (
                batch["image_size"][0] // scale,
                batch["image_size"][1] // scale,
            )
            img_ids_list = [
                self._make_img_ids(
                    latent_size, index=1, h_offset=prompt_len, w_offset=prompt_len
                )
            ]
            if batch["reference_latents"]:
                image_size = batch["reference_sizes"][0]
                reference_latent_size = (
                    image_size[0] // scale,
                    image_size[1] // scale,
                )
                img_ids_list.append(
                    self._make_img_ids(
                        reference_latent_size,
                        index=2,
                        h_offset=prompt_len,
                        w_offset=prompt_len,
                    )
                )
            batch["img_ids"] = torch.cat(img_ids_list, dim=0)

        model_input_list = [batch["noisy_latents"]] + batch["reference_latents"]
        latent_model_input = torch.cat(model_input_list, dim=1)

        model_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return model_pred[:, :n, :]
