from typing import Literal

import torch

from .base import BaseFlux1Adapter


class Flux1KontextAdapter(BaseFlux1Adapter):
    pe_mode: Literal["3d", "diagonal", "stacked"] = "3d"
    pe_index_scale: int = 1

    class BatchType(BaseFlux1Adapter.BatchType):
        reference_latents: list[torch.Tensor]
        """List of `[B, N, D]` Tensors representing VAE encoded reference images."""
        reference_sizes: list[tuple[int, int]]
        """List of `(H, W)` tuples representing the sizes of the reference images."""

    def predict_velocity(
        self, batch: BatchType, timestep: torch.Tensor
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]
        device = batch["noisy_latents"].device
        guidance = torch.full((b,), self.guidance, device=device)

        model_input_list = [batch["noisy_latents"]] + batch["reference_latents"]
        concatenated_model_input = torch.cat(model_input_list, dim=1)
        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            img_ids_list = [self._make_img_ids(batch["image_size"])]
            cur_h = 0
            cur_w = 0
            cur_index = 0
            for size in batch["reference_sizes"]:
                h_ref, w_ref = size
                if self.pe_mode == "3d":
                    cur_index += self.pe_index_scale
                    img_ids_list.append(self._make_img_ids(size, index=cur_index))
                elif self.pe_mode == "diagonal":
                    img_ids_list.append(
                        self._make_img_ids(
                            size,
                            index=0,
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
                            size,
                            index=0,
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
