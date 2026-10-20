from typing import Literal

import torch
from transformers import Qwen2Tokenizer, Qwen3VLModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry, warn_no_image_support


@encoder_registry.register("qwen3vl")
class Qwen3VLEncoder(BaseEncoder[Qwen3VLModel]):
    """Krea 2 text encoder.

    Krea 2 taps 12 intermediate ``Qwen3-VL`` decoder layers and **stacks** them into a
    4D ``(B, seq, num_layers, hidden)`` tensor (NOT concatenated along the feature dim
    like :class:`Qwen3Encoder`); the transformer's internal ``Krea2TextFusion`` collapses
    the layer axis. ``encode_seq_mask`` also returns the bool attention mask, since Krea
    keeps padding tokens in the sequence and passes the mask to the transformer.

    This replicates ``Krea2Pipeline.get_text_hidden_states`` exactly: the Qwen-Image chat
    template with a fixed describe-image system prompt, mid-template padding
    (``[prefix | prompt | PAD | suffix]``), cumulative-valid-token mRoPE positions (so the
    suffix keeps its trained phase), and dropping the system-prefix tokens from the output.
    """

    type: Literal["qwen3vl"] = "qwen3vl"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Qwen3VLModel"
    pretrained_model_id: str = "krea/Krea-2-Raw"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[Qwen2Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id="krea/Krea-2-Raw",
        subfolder="tokenizer",
    )

    max_sequence_length: int = 512
    select_layers: list[int] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    prompt_template_encode_prefix: str = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n"
    )
    prompt_template_encode_suffix: str = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx: int = 34
    prompt_template_encode_num_suffix_tokens: int = 5

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    def encode_seq_mask(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(hidden_states, attention_mask)`` of shapes
        ``(1, seq, num_layers, hidden)`` and ``(1, seq)`` (bool)."""
        tokenizer = self.tokenizer.model
        model = self.model
        device = model.device
        prefix_idx = self.prompt_template_encode_start_idx

        text = [self.prompt_template_encode_prefix + prompt]
        text_tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_sequence_length
            + prefix_idx
            - self.prompt_template_encode_num_suffix_tokens,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = tokenizer(
            [self.prompt_template_encode_suffix], return_tensors="pt"
        ).to(device)

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat(
            [text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1
        ).bool()

        # Krea 2 pads in the middle of the template ([prefix | prompt | PAD | suffix]), so
        # positions must count only real tokens; broadcast across the 3 mRoPE axes
        # (T/H/W are equal for text) as Qwen3-VL expects position_ids of shape (3, B, N).
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in self.select_layers], dim=2
        )

        # Drop the system-prefix tokens from both the features and the mask.
        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt=None):
        return self.encode_seq_mask(prompt)[0]
