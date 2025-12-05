from typing import Any, NotRequired

import torch
from einops import repeat
from pydantic import PrivateAttr

from flow_control.utils.loaders import HfModelLoader

from .base import BaseProcessor


class Flux1Processor(BaseProcessor):
    class BatchType(BaseProcessor.BatchType):
        prompt: str
        negative_prompt: NotRequired[str]
        pooled_prompt_embeds: NotRequired[torch.Tensor]
        prompt_embeds: NotRequired[torch.Tensor]

    _loading_preset = {
        "vae": ["encode", "decode"],
        "text_encoder": ["encode"],
        "text_encoder_2": ["encode"],
        "tokenizer": ["always"],
        "tokenizer_2": ["always"],
    }

    vae: HfModelLoader = HfModelLoader(
        type="diffusers",
        class_name="AutoencoderKL",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="vae",
        dtype=torch.bfloat16,
    )
    _vae: Any = PrivateAttr()

    text_encoder: HfModelLoader = HfModelLoader(
        type="transformers",
        class_name="CLIPTextModel",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
    )
    _text_encoder: Any = PrivateAttr()

    text_encoder_2: HfModelLoader = HfModelLoader(
        type="transformers",
        class_name="T5EncoderModel",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder_2",
        dtype=torch.bfloat16,
    )
    _text_encoder_2: Any = PrivateAttr()

    tokenizer: HfModelLoader = HfModelLoader(
        type="transformers",
        class_name="CLIPTokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer",
    )
    _tokenizer: Any = PrivateAttr()

    tokenizer_2: HfModelLoader = HfModelLoader(
        type="transformers",
        class_name="T5TokenizerFast",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer_2",
    )
    _tokenizer_2: Any = PrivateAttr()

    clip_max_length: int = 77
    t5_max_length: int = 512
    default_negative_prompt: str = "low quality, worst quality, blurry, deformed"

    @torch.no_grad()
    def encode_latents(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode the image into latent space.

        :param image: Input image [B, C, H, W]
        :return: Latent representation [B, C, H', W']
        """

        if image.ndim == 3:
            image = repeat(image, "b h w -> b c h w", c=3)

        image = image * 2 - 1
        image = image.to(torch.bfloat16)
        latent = self._vae.encode(image).latent_dist.sample()
        latent = (
            latent - self._vae.config.shift_factor
        ) * self._vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode the latent representation back to image space.

        :param latents: Latent representation [B, C, H', W']
        :return: Reconstructed image [B, C, H, W]
        """

        latents = (
            latents / self._vae.config.scaling_factor
        ) + self._vae.config.shift_factor
        image = self._vae.decode(latents).sample
        image = (image + 1) / 2
        return image

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the text prompt into embeddings using the two text encoders.

        :param prompt: Input text prompt
        :return: Tuple of embeddings from both text encoders
        """

        # CLIP Text Encoder -> pooled_prompt_embeds
        clip_inputs = self._tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.clip_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        clip_input_ids = clip_inputs.input_ids
        pooled_prompt_embeds = self._text_encoder(
            clip_input_ids.to(self.device), output_hidden_states=False
        ).pooler_output

        # T5 Text Encoder -> prompt_embeds
        t5_inputs = self._tokenizer_2(
            [prompt],
            padding="max_length",
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        t5_input_ids = t5_inputs.input_ids
        prompt_embeds = self._text_encoder_2(
            t5_input_ids.to(self.device), output_hidden_states=False
        )[0]

        return pooled_prompt_embeds, prompt_embeds

    def preprocess_batch(self, batch: BatchType) -> BatchType:
        if "pooled_prompt_embeds" not in batch or "prompt_embeds" not in batch:
            pooled_prompt_embeds, prompt_embeds = self.encode_prompt(batch["prompt"])
            batch["pooled_prompt_embeds"] = pooled_prompt_embeds
            batch["prompt_embeds"] = prompt_embeds
        return batch

    def make_negative_batch(self, batch: BatchType) -> BatchType:
        batch["prompt"] = batch.get("negative_prompt", self.default_negative_prompt)
        return self.preprocess_batch(batch)

    def decode_output(
        self, output_latent: torch.Tensor, batch: BaseProcessor.BatchType
    ) -> torch.Tensor:
        return self.decode_latents(output_latent)
