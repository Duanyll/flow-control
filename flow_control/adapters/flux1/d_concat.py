from typing import Literal

import torch
from diffusers import FluxTransformer2DModel

from ..base import adapter_registry
from .base import Flux1Adapter, Flux1Batch


class Flux1DConcatBatch(Flux1Batch):
    control_latents: torch.Tensor
    """`[B, N, D]` The VAE encoded control condition image."""


@adapter_registry.register("flux1_d_concat")
class Flux1DConcatAdapter(Flux1Adapter[Flux1DConcatBatch]):
    """
    Adapter for applying control to the model through concating the conditional image latent
    to the noisy input latent along the D dimension. It changes the shape of x_embedder layer.

    This is used by Flux.1 Canny and Flux.1 Depth models.
    """

    type: Literal["d_concat"] = "d_concat"
    dense_batch_fields = (*Flux1Adapter.dense_batch_fields, "control_latents")
    input_dimension: int = 128

    lora_target_modules = "all-linear"
    # The widened x_embedder must train whole: its new channels start at zero.
    lora_extra_trainable = (
        "x_embedder",
        "norm_q",
        "norm_k",
        "norm_added_q",
        "norm_added_k",
    )

    def _install_modules(self, transformer: FluxTransformer2DModel) -> None:
        # Change shape of x_embedder layer before loading LoRA
        with torch.no_grad():
            # diffusers' FrozenDict config exposes fields as attributes.
            initial_input_channels: int = transformer.config.in_channels  # ty: ignore[unresolved-attribute]
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

        super()._install_modules(transformer)

    def _predict_velocity(
        self, batch: Flux1DConcatBatch, timestep: torch.Tensor
    ) -> torch.Tensor:
        self._prepare_ids(batch)
        b, n, d = batch["noisy_latents"].shape
        device = batch["noisy_latents"].device
        guidance = torch.full((b,), self.guidance, device=device)

        noisy_model_input = torch.cat(
            (batch["noisy_latents"], batch["control_latents"]), dim=2
        )

        model_pred = self.module()(
            hidden_states=noisy_model_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["_txt_ids"],
            img_ids=batch["_img_ids"],
            return_dict=False,
        )[0]

        return model_pred
