from typing import Literal, NotRequired

import torch

from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)

from .flux1 import Flux1Processor


class KontextProcessor(Flux1Processor):
    class BatchType(Flux1Processor.BatchType):
        reference_images: list[torch.Tensor]
        """List of `[B, C, H, W]` Tensors representing reference images"""
        reference_sizes: NotRequired[list[tuple[int, int]]]
        """List of `(H, W)` tuples representing the sizes of the reference images."""
        reference_latents: NotRequired[list[torch.Tensor]]
        """List of `[B, C, H', W']` Tensors representing VAE encoded reference images."""

    reference_image_resize_mode: Literal["multiple_of", "list", "match_latent"] = (
        "match_latent"
    )

    def preprocess_batch(self, batch: BatchType):
        if "pooled_prompt_embeds" not in batch or "prompt_embeds" not in batch:
            pooled_prompt_embeds, prompt_embeds = self.encode_prompt(batch["prompt"])
            batch["pooled_prompt_embeds"] = pooled_prompt_embeds
            batch["prompt_embeds"] = prompt_embeds

        if "clean_image" in batch and "clean_latents" not in batch:
            batch["clean_image"] = self.resize_image(batch["clean_image"])
            batch["image_size"] = (
                batch["clean_image"].shape[2],
                batch["clean_image"].shape[3],
            )
            batch["clean_latents"] = self.encode_latents(batch["clean_image"])

        if "image_size" not in batch:
            if len(batch["reference_images"]) > 0:
                batch["reference_images"][0] = self.resize_image(
                    batch["reference_images"][0]
                )
                batch["image_size"] = (
                    batch["reference_images"][0].shape[2],
                    batch["reference_images"][0].shape[3],
                )
            else:
                batch["image_size"] = self.default_resolution

        if "reference_latents" not in batch:
            reference_latents = []
            reference_sizes = []
            for img in batch["reference_images"]:
                img = img.to(self.device)
                if self.reference_image_resize_mode == "multiple_of":
                    img = resize_to_multiple_of(img, multiple=self.multiple_of)
                elif self.reference_image_resize_mode == "list":
                    img = resize_to_closest_resolution(img, self.preferred_resolutions)
                else:  # match_latent
                    img = resize_to_resolution(img, batch["image_size"])
                reference_sizes.append((img.shape[2], img.shape[3]))
                latents = self.encode_latents(img)
                reference_latents.append(latents)
            batch["reference_latents"] = reference_latents
            batch["reference_sizes"] = reference_sizes

        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        batch["latent_length"] = (
            batch["prompt_embeds"].shape[1]
            + batch["image_size"][0] * batch["image_size"][1] // ratio
            + sum(
                latents.shape[2] * latents.shape[3] // ratio
                for latents in batch["reference_latents"]
            )
        )

        return batch
