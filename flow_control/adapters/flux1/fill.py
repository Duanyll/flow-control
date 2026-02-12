import torch
from einops import pack, rearrange

from .base import Flux1Adapter, Flux1Batch


class Flux1FillBatch(Flux1Batch):
    inpaint_latents: torch.Tensor
    """`[B, N, D]` The latents of the inpainted image."""
    inpaint_mask: torch.Tensor
    """`[B, 1, H, W]` The inpainting mask. Can be a boolean tensor or a tuple of
        (indices, values) for sparse representation."""


class Flux1FillAdapter(Flux1Adapter[Flux1FillBatch]):
    """
    Adapter for the FLUX.1 fill model.
    """

    enforce_mask: bool = False

    def predict_velocity(
        self,
        batch: Flux1FillBatch,
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

        mask = self._pack_mask(batch["inpaint_mask"])
        inputs = pack(
            [
                batch["noisy_latents"],
                batch["inpaint_latents"],
                self._pack_latents(mask),
            ],
            "b n *",
        )

        model_pred = self.transformer(
            hidden_states=inputs,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return model_pred

    def _pack_mask(self, mask) -> torch.Tensor:
        """
        Pack the mask with PixelShuffle-like operation, as seen in Flux.1 Fill
        """
        mask = mask.to(torch.bfloat16)
        return rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
