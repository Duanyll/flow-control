import math
from typing import Any, Literal

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry, warn_no_image_support


@encoder_registry.register("flux3")
class Flux3Encoder(BaseEncoder[Qwen3VLForConditionalGeneration]):
    """FLUX 3 Image Klein text encoder: the official ``text_encoder.py`` recipe.

    Qwen3-VL-4B-Instruct over the plain user-turn chat template; eight hidden
    layers are concatenated per token into ``(1, L, 20480)`` with ``L`` right-padded
    to a multiple of 80. The padding is masked inside Qwen3-VL (position id 1, as
    in the training environment) but kept in the output without a mask: the DiT
    attends to it.
    """

    type: Literal["flux3"] = "flux3"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Qwen3VLForConditionalGeneration"
    pretrained_model_id: str = "black-forest-labs/flux-3-image-klein"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[Qwen3VLProcessor] = HfModelLoader(
        library="transformers",
        class_name="Qwen3VLProcessor",
        pretrained_model_id="black-forest-labs/flux-3-image-klein",
        subfolder="text_encoder",
    )

    max_sequence_length: int = 4096
    pad_multiple: int = 80
    hidden_state_layers: list[int] = [4, 8, 12, 16, 20, 24, 28, 32]

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt=None):
        processor: Any = self.tokenizer.model
        text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenizer = processor.tokenizer
        real = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_sequence_length,
        )["input_ids"].shape[1]
        length = min(
            math.ceil(real / self.pad_multiple) * self.pad_multiple,
            self.max_sequence_length,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=length,
        )
        device = self.model.device
        mask = inputs["attention_mask"].to(device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        output = self.model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
        return torch.cat(
            [output.hidden_states[k] for k in self.hidden_state_layers], dim=-1
        ).to(self.dtype)
