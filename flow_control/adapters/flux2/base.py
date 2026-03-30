from typing import Literal, NotRequired

import torch
from diffusers import Flux2Transformer2DModel
from einops import rearrange

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class Flux2Batch(Batch):
    prompt_embeds: torch.Tensor
    """`[B, N, D]` Text embeddings from the Mistral3 / Qwen3 text encoder."""
    reference_latents: list[torch.Tensor]
    """List of `[B, N, D]` Tensors representing VAE encoded reference images."""
    reference_sizes: list[tuple[int, int]]
    """List of `(H, W)` tuples representing the sizes of the reference images."""
    txt_ids: NotRequired[torch.Tensor]
    """`[B, N, 4]` Used for adding positional embeddings to the text embeddings."""
    img_ids: NotRequired[torch.Tensor]
    """`[B, N, 4]` Used for adding positional embeddings to the image embeddings."""


class Flux2Adapter[TBatch: Flux2Batch](
    BaseModelAdapter[Flux2Transformer2DModel, TBatch]
):
    arch: Literal["flux2"] = "flux2"
    type: Literal["base"] = "base"

    hf_model: HfModelLoader[Flux2Transformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="Flux2Transformer2DModel",
        pretrained_model_id="black-forest-labs/FLUX.2-dev",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    patch_size: int = 2
    vae_scale_factor: int = 8
    latent_channels: int = 32

    guidance: float = 4.0
    time_scale: int = 10

    def make_text_ids(
        self, prompt_embeds: torch.Tensor, t_coord: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, _ = prompt_embeds.shape
        out_ids = []

        for i in range(b):
            t = torch.arange(1) if t_coord is None else t_coord[i : i + 1]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(n)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        out_ids = torch.stack(out_ids, dim=0)
        return out_ids.to(prompt_embeds.device)

    def make_latent_ids(self, image_size: tuple[int, int]) -> torch.Tensor:
        height, width = image_size
        height = height // self.vae_scale_factor // self.patch_size
        width = width // self.vae_scale_factor // self.patch_size

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        ids = torch.cartesian_prod(t, h, w, l)
        ids = rearrange(ids, "n d -> 1 n d")
        return ids.to(self.device)

    def make_reference_ids(
        self, reference_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        # t_coords: [10, 20, 30, ...] for up to 10 reference images
        t_coords = torch.arange(1, len(reference_sizes) + 1) * self.time_scale
        reference_ids = []
        for size, t in zip(reference_sizes, t_coords, strict=False):
            height, width = size
            height = height // self.vae_scale_factor // self.patch_size
            width = width // self.vae_scale_factor // self.patch_size

            h = torch.arange(height)
            w = torch.arange(width)
            l = torch.arange(1)

            coords = torch.cartesian_prod(t[None], h, w, l)
            reference_ids.append(coords)
        reference_ids = torch.cat(reference_ids, dim=0)
        reference_ids = rearrange(reference_ids, "n d -> 1 n d")
        return reference_ids.to(self.device)

    def make_guidance(self) -> torch.Tensor | None:
        if self.transformer.config["guidance_embeds"]:
            return torch.tensor([self.guidance], device=self.device, dtype=self.dtype)
        else:
            return None

    def _predict_velocity(self, batch, timestep):
        b, n, d = batch["noisy_latents"].shape
        guidance = self.make_guidance()
        if "reference_latents" in batch and "reference_sizes" in batch:
            latent_model_input = torch.cat(
                [batch["noisy_latents"]] + batch["reference_latents"], dim=1
            )
            if "img_ids" not in batch:
                batch["img_ids"] = torch.cat(
                    [self.make_latent_ids(batch["image_size"])]
                    + [self.make_reference_ids(batch["reference_sizes"])],
                    dim=1,
                )
            img_ids = batch["img_ids"]
        else:
            latent_model_input = batch["noisy_latents"]
            if "img_ids" not in batch:
                batch["img_ids"] = self.make_latent_ids(batch["image_size"])
            img_ids = batch["img_ids"]

        if "txt_ids" not in batch:
            batch["txt_ids"] = self.make_text_ids(batch["prompt_embeds"])
        txt_ids = batch["txt_ids"]

        model_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

        return model_pred[:, :n, :]
