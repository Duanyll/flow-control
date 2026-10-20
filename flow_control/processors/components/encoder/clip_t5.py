from typing import Literal

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry, warn_no_image_support


@encoder_registry.register("t5")
class T5TextEncoder(BaseEncoder[T5EncoderModel]):
    type: Literal["t5"] = "t5"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "T5EncoderModel"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "text_encoder_2"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[T5Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="T5Tokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer_2",
    )

    max_length: int | None = None
    """Cap the padded sequence length. When ``None`` the tokenizer's own
    ``model_max_length`` is used (FLUX behaviour). SD3 sets this to 256."""

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        fresh = super().load_model(device, frozen)
        # Transformers 5.17 made SDPA the default for T5. In bf16 it moves the
        # FLUX/SD3 T5-XXL embeddings by up to 20% (relative L2) and lands further
        # from an fp32 run than eager does; eager matches transformers <= 5.16
        # bitwise, which every existing T5 embedding was computed with.
        self.model.set_attn_implementation("eager")
        return fresh

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt: str | None = None):
        tokenizer = self.tokenizer.model
        model = self.model

        prompt = self._format_prompt(prompt, images, system_prompt)

        t5_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        t5_input_ids = t5_inputs.input_ids
        prompt_embeds = model(
            t5_input_ids.to(model.device), output_hidden_states=False
        )[0]

        return prompt_embeds


@encoder_registry.register("clip")
class ClipTextEncoder(BaseEncoder[CLIPTextModel]):
    type: Literal["clip"] = "clip"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "CLIPTextModel"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[CLIPTokenizer] = HfModelLoader(
        library="transformers",
        class_name="CLIPTokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer",
    )

    max_length: int = 77

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt: str | None = None):
        tokenizer = self.tokenizer.model
        model = self.model

        prompt = self._format_prompt(prompt, images, system_prompt)

        clip_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        clip_input_ids = clip_inputs.input_ids
        pooled_prompt_embeds = model(
            clip_input_ids.to(model.device), output_hidden_states=False
        ).pooler_output

        return pooled_prompt_embeds


if __name__ == "__main__":
    from rich import print
    from transformers import AttentionInterface

    from flow_control.utils import device as devutil

    # T5 (FLUX.1, SD3): bitwise equal to the attention math of transformers
    # <= 5.16 (bf16 scores + position bias, fp32 softmax).
    def legacy_t5_attention(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: None,  # unregistered with the mask interface
        position_bias: torch.Tensor,
        **kwargs,
    ):
        scores = torch.matmul(query, key.transpose(3, 2))
        scores += position_bias
        weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights

    AttentionInterface.register("t5_legacy", legacy_t5_attention)
    t5 = T5TextEncoder()
    t5.load_model(devutil.default_device())
    prompt = 'A neon sign that says "OPEN 24/7" above a rainy street'
    ours = t5.encode(prompt)
    t5.model.set_attn_implementation("t5_legacy")
    reference = t5.encode(prompt)
    assert torch.equal(ours, reference), "T5 no longer matches transformers <= 5.16"
    print("[green]T5: bitwise equal to the transformers <= 5.16 attention[/]")
    t5.unload_model()
