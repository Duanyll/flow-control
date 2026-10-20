from typing import Literal

import torch
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry, warn_no_image_support


@encoder_registry.register("qwen3")
class Qwen3Encoder(BaseEncoder[Qwen3ForCausalLM]):
    type: Literal["qwen3"] = "qwen3"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Qwen3ForCausalLM"
    pretrained_model_id: str = "Tongyi-MAI/Z-Image"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[Qwen2Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id="Tongyi-MAI/Z-Image",
        subfolder="tokenizer",
    )

    max_sequence_length: int = 512
    hidden_state_layers: list[int] = [-2]
    enable_thinking: bool = True
    keep_padding_tokens: bool = False

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt=None):
        messages = [{"role": "user", "content": prompt}]
        formated_prompt = self.tokenizer.model.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        assert isinstance(formated_prompt, str)
        text_inputs = self.tokenizer.model(
            [formated_prompt],
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.model.device)
        prompt_masks = text_inputs.attention_mask.to(self.model.device).bool()
        hidden_states = self.model(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states
        prompt_embeds = torch.cat(
            [hidden_states[i] for i in self.hidden_state_layers], dim=2
        )
        if not self.keep_padding_tokens:
            prompt_embeds = prompt_embeds[prompt_masks].unsqueeze(0)
        return prompt_embeds
