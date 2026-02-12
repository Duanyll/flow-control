from typing import Literal

import torch

from .base import Flux1Adapter, Flux1Batch


class Flux1KontextBatch(Flux1Batch):
    reference_latents: list[torch.Tensor]
    """List of `[B, N, D]` Tensors representing VAE encoded reference images."""
    reference_sizes: list[tuple[int, int]]
    """List of `(H, W)` tuples representing the sizes of the reference images."""


class Flux1KontextAdapter(Flux1Adapter[Flux1KontextBatch]):
    pe_mode: Literal["3d", "diagonal", "stacked"] = "3d"
    pe_index_scale: int = 1

    def predict_velocity(
        self, batch: Flux1KontextBatch, timestep: torch.Tensor
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]
        device = batch["noisy_latents"].device
        guidance = (
            torch.full((b,), self.guidance, device=device)
            if self.guidance is not None
            else None
        )

        model_input_list = [batch["noisy_latents"]] + batch["reference_latents"]
        concatenated_model_input = torch.cat(model_input_list, dim=1)
        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            scale = self.patch_size * self.vae_scale_factor
            latent_size = (
                batch["image_size"][0] // scale,
                batch["image_size"][1] // scale,
            )
            img_ids_list = [self._make_img_ids(latent_size)]
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
            batch["img_ids"] = torch.cat(img_ids_list, dim=0)

        model_pred = self.transformer(
            hidden_states=concatenated_model_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return model_pred[:, :n, :]
