import torch
from einops import repeat

from .peft_lora import Flux1PeftLoraAdapter


class Flux1NConcatAdapter(Flux1PeftLoraAdapter):
    """
    Adapter for applying control to the model through concatenating the conditional image latent
    to the noisy input latent along the N dimension.

    This is used by the PhotoDoddle model.
    """

    class BatchType(Flux1PeftLoraAdapter.BatchType):
        control_latents: torch.Tensor
        """`[B, N, D]` The VAE encoded control condition image."""

    def predict_velocity(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = (
            torch.full((b,), self.guidance, device=device)
            if self.guidance is not None
            else None
        )

        noisy_model_input = batch["noisy_latents"]
        control_model_input = batch["control_latents"]
        concatenated_model_input = torch.cat(
            (noisy_model_input, control_model_input), dim=1
        )
        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            batch["img_ids"] = repeat(
                self._make_img_ids(batch["image_size"]), "n d -> (r n) d", r=2
            )

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

        model_pred = model_pred[:, :n, :]

        return model_pred
