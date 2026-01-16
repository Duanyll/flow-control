from typing import Literal, NotRequired

import torch

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)

from .qwen import QwenImageProcessor


class QwenImageEditProcessor(QwenImageProcessor):
    class BatchType(QwenImageProcessor.BatchType):
        reference_images: list[torch.Tensor]
        """List of `[B, C, H, W]` Tensors representing reference images"""
        reference_sizes: NotRequired[list[tuple[int, int]]]
        """List of `(H, W)` tuples representing the sizes of the reference images."""
        reference_latents: NotRequired[list[torch.Tensor]]
        """List of `[B, C, H', W']` Tensors representing VAE encoded reference images."""

    _encoding_components = ["vae"]
    _decoding_components = ["vae", "text_encoder", "tokenizer", "vl_processor"]

    vl_processor: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="Qwen2VLProcessor",
        pretrained_model_id="Qwen/Qwen-Image-Edit",
        subfolder="processor",
    )

    reference_image_resize_mode: Literal["multiple_of", "list", "match_latent"] = (
        "match_latent"
    )

    prompt_template_encode: str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx: int = 64
    image_prompt_template: str = (
        "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    )

    @torch.no_grad()
    def encode_prompt_with_images(
        self, prompt: str, reference_images: list[torch.Tensor]
    ):
        txt = self.prompt_template_encode.format(
            "".join(
                [
                    self.image_prompt_template.format(i + 1)
                    for i in range(len(reference_images))
                ]
                + [prompt]
            )
        )
        model_inputs = self.vl_processor.model(
            text=txt,
            images=reference_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        encoder_hidden_states = self.text_encoder.model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        drop_idx = self.prompt_template_encode_start_idx
        return hidden_states[:, drop_idx:, :]

    def preprocess_batch(self, batch: BatchType) -> BatchType:
        if "prompt_embeds" not in batch:
            prompt_embeds = self.encode_prompt_with_images(
                batch["prompt"], batch["reference_images"]
            )
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
