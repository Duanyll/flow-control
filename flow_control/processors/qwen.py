from typing import Literal, NotRequired

import torch
from einops import repeat

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.resize import (
    ResolutionList,
    resize_to_closest_resolution,
    resize_to_multiple_of,
)
from flow_control.utils.vae import QwenImageVAE

from .base import BaseProcessor


class QwenImageProcessor(BaseProcessor):
    class BatchType(BaseProcessor.BatchType):
        prompt: str
        negative_prompt: NotRequired[str]
        prompt_embeds: NotRequired[torch.Tensor]
        clean_image: NotRequired[torch.Tensor]
        clean_latents: NotRequired[torch.Tensor]

    _encoding_components = ["vae"]
    _decoding_components = ["vae", "text_encoder", "tokenizer"]

    vae: QwenImageVAE = QwenImageVAE()

    text_encoder: HfModelLoader = HfModelLoader(
        type="transformers",
        class_name="Qwen2_5_VLForConditionalGeneration",
        pretrained_model_id="Qwen/Qwen-Image",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
    )

    tokenizer: HfModelLoader = HfModelLoader(
        type="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id="Qwen/Qwen-Image",
        subfolder="tokenizer",
    )

    max_sequence_length: int = 512
    default_negative_prompt: str = "low quality, worst quality, blurry, deformed"

    default_resolution: tuple[int, int] = (1328, 1328)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = [
        (1328, 1328),
        (1664, 928),
        (928, 1664),
        (1472, 1104),
        (1104, 1472),
        (1584, 1056),
        (1056, 1584),
    ]
    multiple_of: int = 32
    pixels: int = 1328 * 1328

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.resize_mode == "list":
            return resize_to_closest_resolution(
                image,
                self.preferred_resolutions,
                crop=True,
            )
        else:
            return resize_to_multiple_of(
                image, self.multiple_of, crop=True, pixels=self.pixels
            )

    @torch.no_grad()
    def encode_latents(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode the image into latent space.

        :param image: Input image [B, C, H, W]
        :return: Latent representation [B, C, H', W']
        """

        if image.ndim == 3:
            image = repeat(image, "b h w -> b c h w", c=3)
        latents = self.vae.encode(image)
        latents = self._pack_latents(latents)
        return latents

    @torch.no_grad()
    def decode_latents(
        self, latents: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Decode the latent representation back to image space.

        :param latents: Latent representation [B, C, H', W']
        :return: Reconstructed image [B, C, H, W]
        """
        latents = self._unpack_latents(latents, size)
        image = self.vae.decode(latents)
        return image

    tokenizer_max_length: int = 1024
    prompt_template_encode: str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx: int = 34

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        """
        Encode the text prompt into embeddings using the two text encoders.

        :param prompt: Input text prompt
        :return: Tuple of embeddings from both text encoders
        """

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(prompt)]
        txt_tokens = self.tokenizer.model(
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        encoder_hidden_states = self.text_encoder.model(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        return hidden_states[:, drop_idx:, :]

    def preprocess_batch(self, batch: BatchType) -> BatchType:
        if "prompt_embeds" not in batch:
            prompt_embeds = self.encode_prompt(batch["prompt"])
            batch["prompt_embeds"] = prompt_embeds

        if "clean_image" in batch and "clean_latents" not in batch:
            batch["clean_image"] = self.resize_image(batch["clean_image"])
            batch["image_size"] = (
                batch["clean_image"].shape[2],
                batch["clean_image"].shape[3],
            )
            batch["clean_latents"] = self.encode_latents(batch["clean_image"])

        if "image_size" not in batch:
            batch["image_size"] = self.default_resolution

        latent_h = batch["image_size"][0] // self.vae_scale_factor // self.patch_size
        latent_w = batch["image_size"][1] // self.vae_scale_factor // self.patch_size
        batch["latent_length"] = batch["prompt_embeds"].shape[1] + latent_h * latent_w
        return batch

    def make_negative_batch(self, batch: BatchType) -> BatchType:
        batch["prompt"] = batch.get("negative_prompt", self.default_negative_prompt)
        del batch["negative_prompt"]
        del batch["prompt_embeds"]
        return self.preprocess_batch(batch)

    def decode_output(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> torch.Tensor:
        batch["clean_image"] = self.decode_latents(output_latent, batch["image_size"])  # type: ignore
        return batch["clean_image"]
