import torch
from peft import LoraConfig

from .base import Flux1Adapter, Flux1Batch


class Flux1DConcatBatch(Flux1Batch):
    control_latents: torch.Tensor
    """`[B, N, D]` The VAE encoded control condition image."""


class Flux1DConcatAdapter(Flux1Adapter[Flux1DConcatBatch]):
    """
    Adapter for applying control to the model through concating the conditional image latent
    to the noisy input latent along the D dimension. It changes the shape of x_embedder layer.

    This is used by Flux.1 Canny and Flux.1 Depth models.
    """

    input_dimension: int = 128

    peft_lora_rank: int = 128
    peft_lora_config: LoraConfig = LoraConfig(
        target_modules="all-linear", lora_bias=True
    )
    extra_trainable_modules: list[str] = [
        "x_embedder",
        "norm_q",
        "norm_k",
        "norm_added_q",
        "norm_added_k",
    ]

    def _install_modules(self):
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

        super()._install_modules()

    def predict_velocity(
        self, batch: Flux1DConcatBatch, timestep: torch.Tensor
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = (
            torch.full((b,), self.guidance, device=device)
            if self.guidance is not None
            else None
        )

        noisy_model_input = torch.cat(
            (batch["noisy_latents"], batch["control_latents"]), dim=2
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

        return model_pred
