from typing import Any, Literal

import torch
from transformers import Qwen3VLProcessor

from flow_control.utils.resize import resize_to_resolution
from flow_control.utils.tensor import remove_alpha_channel
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry


@encoder_registry.register("hidream_o1")
class HiDreamO1Encoder(BaseEncoder[Qwen3VLProcessor]):
    """Tokenizer-only "encoder" for HiDream-O1 (pixel-space unified transformer).

    HiDream-O1 has no separate text encoder: prompt token ids are consumed by the
    same trainable transformer that denoises pixels, so encoding a prompt is pure
    CPU tokenization. ``self.model`` is the HF ``Qwen3VLProcessor`` (tokenizer +
    image processor); no GPU model is ever loaded.

    - :meth:`encode_ids` reproduces the official ``build_t2i_text_sample`` text
      part: ``chat_template(prompt) + <|boi_token|> + <|tms_token|>*n``. The
      trailing target-image vision tokens are resolution-dependent and appended
      by the adapter at forward time.
    - :meth:`encode_ids_with_images` reproduces the editing/personalization
      conditioning: reference thumbnails go through the model's own SigLIP tower
      via ``pixel_values``/``image_grid_thw`` with image placeholders in the chat
      template (official ``CONDITION_IMAGE_SIZE`` K-count heuristic).
    """

    type: Literal["hidream_o1"] = "hidream_o1"
    library: Literal["diffusers", "transformers", "custom"] = "transformers"
    class_name: str = "Qwen3VLProcessor"
    pretrained_model_id: str = "HiDream-ai/HiDream-O1-Image"
    subfolder: str | None = None
    dtype: TorchDType = torch.bfloat16

    boi_token: str = "<|boi_token|>"
    tms_token: str = "<|tms_token|>"
    num_timestep_tokens: int = 1
    condition_image_size: int = 384
    """Base thumbnail size for the SigLIP condition path; scaled down with the
    reference count K like the official pipeline (K<=4: 384, K<=8: 288, else 192)."""

    def _apply_chat_template(
        self, content: Any, system_prompt: str | None = None
    ) -> str:
        processor: Any = self.model
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        template = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return template + self.boi_token + self.tms_token * self.num_timestep_tokens

    def encode_ids(self, prompt: str, system_prompt: str | None = None) -> torch.Tensor:
        """Tokenize a text-only prompt; returns ``input_ids`` of shape ``[1, L]``."""
        template = self._apply_chat_template(prompt, system_prompt)
        tokenizer: Any = self.model.tokenizer
        return tokenizer.encode(template, return_tensors="pt", add_special_tokens=False)

    def _thumbnail_size(self, num_images: int) -> int:
        if num_images <= 4:
            return self.condition_image_size
        if num_images <= 8:
            return self.condition_image_size * 48 // 64
        return self.condition_image_size // 2

    def encode_ids_with_images(
        self,
        prompt: str,
        images: list[torch.Tensor],
        system_prompt: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a prompt with K reference-image placeholders and preprocess
        the SigLIP-condition thumbnails.

        Returns ``{"input_ids": [1, L], "pixel_values", "image_grid_thw"}``.
        Input images are BCHW in ``[0, 1]``; ``do_rescale=False`` avoids the
        processor's unconditional ``1/255`` rescale (it would double-rescale).
        """
        from flow_control.third_party.hidream_o1 import calculate_dimensions

        cond_size = self._thumbnail_size(len(images))
        thumbnails = []
        for image in images:
            image = remove_alpha_channel(image)
            width, height = calculate_dimensions(
                cond_size, image.shape[3] / image.shape[2]
            )
            thumbnails.append(resize_to_resolution(image, (height, width)))

        content: list[dict[str, Any]] = [{"type": "image"} for _ in thumbnails]
        content.append({"type": "text", "text": prompt})
        template = self._apply_chat_template(content, system_prompt)

        processor: Any = self.model
        proc = processor(
            text=[template],
            images=thumbnails,
            text_kwargs={"padding": "longest", "return_tensors": "pt"},
            images_kwargs={"do_rescale": False, "return_tensors": "pt"},
        )
        return {
            "input_ids": proc.input_ids,
            "pixel_values": proc.pixel_values,
            "image_grid_thw": proc.image_grid_thw,
        }

    def encode(self, prompt, images=None, system_prompt=None):
        if images:
            return self.encode_ids_with_images(prompt, images, system_prompt)[
                "input_ids"
            ]
        return self.encode_ids(prompt, system_prompt)
