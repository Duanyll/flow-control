from typing import Literal, NotRequired

import torch
from einops import repeat

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.resize import (
    ResolutionList,
    resize_to_closest_resolution,
    resize_to_multiple_of,
)

from .base import BaseProcessor
from .components.vae import VAE, Flux1VAE


class Flux1Processor(BaseProcessor):
    class BatchType(BaseProcessor.BatchType):
        prompt: str
        negative_prompt: NotRequired[str]
        pooled_prompt_embeds: NotRequired[torch.Tensor]
        prompt_embeds: NotRequired[torch.Tensor]
        clean_image: NotRequired[torch.Tensor]
        clean_latents: NotRequired[torch.Tensor]

    _encoding_components = [
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
    ]
    _decoding_components = ["vae"]

    vae: VAE = Flux1VAE()

    text_encoder: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="CLIPTextModel",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
    )

    text_encoder_2: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="T5EncoderModel",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder_2",
        dtype=torch.bfloat16,
    )

    tokenizer: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="CLIPTokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer",
    )

    tokenizer_2: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="T5TokenizerFast",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer_2",
    )

    clip_max_length: int = 77
    t5_max_length: int = 512
    default_negative_prompt: str = "low quality, worst quality, blurry, deformed"

    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = [
        (672, 1568),
        (688, 1504),
        (720, 1456),
        (752, 1392),
        (800, 1328),
        (832, 1248),
        (880, 1184),
        (944, 1104),
        (1024, 1024),
        (1104, 944),
        (1184, 880),
        (1248, 832),
        (1328, 800),
        (1392, 752),
        (1456, 720),
        (1504, 688),
        (1568, 672),
    ]
    multiple_of: int = 32

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.resize_mode == "list":
            return resize_to_closest_resolution(
                image, self.preferred_resolutions, crop=True
            )
        else:
            return resize_to_multiple_of(image, self.multiple_of, crop=True)

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

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the text prompt into embeddings using the two text encoders.

        :param prompt: Input text prompt
        :return: Tuple of embeddings from both text encoders
        """

        # CLIP Text Encoder -> pooled_prompt_embeds
        clip_inputs = self.tokenizer.model(
            [prompt],
            padding="max_length",
            max_length=self.clip_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        clip_input_ids = clip_inputs.input_ids
        pooled_prompt_embeds = self.text_encoder.model(
            clip_input_ids.to(self.device), output_hidden_states=False
        ).pooler_output

        # T5 Text Encoder -> prompt_embeds
        t5_inputs = self.tokenizer_2.model(
            [prompt],
            padding="max_length",
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        t5_input_ids = t5_inputs.input_ids
        prompt_embeds = self.text_encoder_2.model(
            t5_input_ids.to(self.device), output_hidden_states=False
        )[0]

        return pooled_prompt_embeds, prompt_embeds

    def preprocess_batch(self, batch: BatchType) -> BatchType:
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
            batch["image_size"] = self.default_resolution

        latent_h = batch["image_size"][0] // self.vae_scale_factor // self.patch_size
        latent_w = batch["image_size"][1] // self.vae_scale_factor // self.patch_size
        batch["latent_length"] = batch["prompt_embeds"].shape[1] + latent_h * latent_w
        return batch

    def make_negative_batch(self, batch: BatchType) -> BatchType:
        batch["prompt"] = batch.get("negative_prompt", self.default_negative_prompt)
        del batch["negative_prompt"]
        del batch["prompt_embeds"]
        del batch["pooled_prompt_embeds"]
        return self.preprocess_batch(batch)

    def decode_output(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> torch.Tensor:
        batch["clean_image"] = self.decode_latents(output_latent, batch["image_size"])  # type: ignore
        return batch["clean_image"]
