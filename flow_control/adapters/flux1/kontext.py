from typing import Literal

import torch
from diffusers import FluxTransformer2DModel

from flow_control.utils.hf_model import HfModelLoader

from ..base import adapter_registry
from .base import Flux1Adapter, Flux1Batch


class Flux1KontextBatch(Flux1Batch):
    reference_latents: list[torch.Tensor]
    """List of `[B, N, D]` Tensors representing VAE encoded reference images."""
    reference_sizes: list[tuple[int, int]]
    """List of `(H, W)` tuples representing the sizes of the reference images."""


@adapter_registry.register("flux1_kontext")
class Flux1KontextAdapter(Flux1Adapter[Flux1KontextBatch]):
    type: Literal["kontext"] = "kontext"
    dense_batch_fields = (
        *Flux1Adapter.dense_batch_fields,
        "reference_latents",
        "reference_sizes",
    )
    hf_model: HfModelLoader[FluxTransformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="FluxTransformer2DModel",
        pretrained_model_id="black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    pe_mode: Literal["3d", "diagonal", "stacked"] = "3d"
    pe_index_scale: int = 1

    def _predict_velocity(
        self, batch: Flux1KontextBatch, timestep: torch.Tensor
    ) -> torch.Tensor:
        self._prepare_ids(batch)
        b, n, d = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = torch.full((b,), self.guidance, device=device)

        model_input_list = [batch["noisy_latents"]] + batch["reference_latents"]
        concatenated_model_input = torch.cat(model_input_list, dim=1)
        model_pred = self.transformer(
            hidden_states=concatenated_model_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["_txt_ids"],
            img_ids=batch["_img_ids"],
            return_dict=False,
        )[0]

        return model_pred[:, :n, :]

    def _make_batch_img_ids(self, batch: Flux1KontextBatch) -> torch.Tensor:
        h, w = batch["image_size"]
        scale = self.patch_size * self.vae_scale_factor
        img_ids_list = [super()._make_batch_img_ids(batch)]
        cur_h = 0
        cur_w = 0
        cur_index = 0
        for size in batch["reference_sizes"]:
            h_ref, w_ref = lsize = (size[0] // scale, size[1] // scale)
            cur_index += self.pe_index_scale
            if self.pe_mode == "3d":
                img_ids_list.append(self._make_img_ids(lsize, index=cur_index))
            elif self.pe_mode == "diagonal":
                img_ids_list.append(
                    self._make_img_ids(
                        lsize,
                        index=cur_index,
                        h_offset=h + cur_h,
                        w_offset=w + cur_w,
                    )
                )
                cur_h += h_ref
                cur_w += w_ref
            elif self.pe_mode == "stacked":
                h_offset = 0
                w_offset = 0
                if h_ref + cur_h > w_ref + cur_w:
                    w_offset = cur_w
                else:
                    h_offset = cur_h
                cur_h = max(cur_h, h_ref + h_offset)
                cur_w = max(cur_w, w_ref + w_offset)
                img_ids_list.append(
                    self._make_img_ids(
                        lsize,
                        index=cur_index,
                        h_offset=h_offset,
                        w_offset=w_offset,
                    )
                )
        return torch.cat(img_ids_list, dim=0)
