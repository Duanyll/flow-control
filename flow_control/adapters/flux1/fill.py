import torch
from einops import pack, rearrange

from .base import BaseFlux1Adapter


class Flux1FillAdapter(BaseFlux1Adapter):
    """
    Adapter for the FLUX.1 fill model.
    """

    enforce_mask: bool = False

    class BatchType(BaseFlux1Adapter.BatchType):
        inpaint_latents: torch.Tensor
        """`[B, C, H, W]` The latents of the inpainted image."""
        inpaint_mask: torch.Tensor
        """`[B, 1, H, W]` The inpainting mask. Can be a boolean tensor or a tuple of
        (indices, values) for sparse representation."""

    def predict_velocity(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = torch.full((b,), self.guidance, device=device)

        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])

        mask = self._pack_mask(batch["inpaint_mask"])
        inputs = pack(
            [
                self._pack_latents(batch["noisy_latents"]),
                self._pack_latents(batch["inpaint_latents"]),
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

        velocity = self._unpack_latents(model_pred, h, w)
        if self.enforce_mask:
            velocity = velocity.to(torch.float32)
            target_velocity = (
                batch["inpaint_latents"].float() - batch["noisy_latents"].float()
            ) / timestep
            velocity = velocity * mask + target_velocity * (1 - mask)
        return velocity

    def _pack_mask(self, mask) -> torch.Tensor:
        """
        Pack the mask with PixelShuffle-like operation, as seen in Flux.1 Fill
        """
        mask = mask.to(torch.bfloat16)
        return rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
