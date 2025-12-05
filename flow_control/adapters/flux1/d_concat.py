from typing import Literal

import torch

from flow_control.utils.common import ensure_trainable

from .peft_lora import Flux1PeftLoraAdapter


class Flux1DConcatAdapter(Flux1PeftLoraAdapter):
    """
    Adapter for applying control to the model through concating the conditional image latent
    to the noisy input latent along the D dimension. It changes the shape of x_embedder layer.

    This is used by Flux.1 Canny and Flux.1 Depth models.
    """

    train_norm_layers: bool = True
    lora_layers: Literal["all-linear"] | list[str] = "all-linear"
    rank = 128
    use_lora_bias: bool = True
    input_dimension = 128

    class BatchType(Flux1PeftLoraAdapter.BatchType):
        control_latents: torch.Tensor
        """`[B, C, H, W]` The VAE encoded control condition image."""

    def install_modules(self):
        transformer = self.transformer
        # Change shape of x_embedder layer before loading LoRA
        with torch.no_grad():
            initial_input_channels: int = transformer.config.in_channels  # type: ignore
            new_linear = torch.nn.Linear(
                self.input_dimension,
                transformer.x_embedder.out_features,
                bias=transformer.x_embedder.bias is not None,
                dtype=transformer.dtype,
                device=transformer.device,
            )
            new_linear.weight.zero_()
            new_linear.weight[:, :initial_input_channels].copy_(
                transformer.x_embedder.weight
            )
            if transformer.x_embedder.bias is not None:
                new_linear.bias.copy_(transformer.x_embedder.bias)
            transformer.x_embedder = new_linear

        super().install_modules()
        ensure_trainable(transformer.x_embedder)

    def predict_velocity(self, batch: dict, timestep: torch.Tensor) -> torch.Tensor:
        b, c, h, w = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = torch.full((b,), self.guidance, device=device)

        noisy_model_input = torch.cat(
            (batch["noisy_latents"], batch["control_latents"]), dim=1
        )

        if "txt_ids" not in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if "img_ids" not in batch:
            batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])

        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return self._unpack_latents(model_pred, h, w)
