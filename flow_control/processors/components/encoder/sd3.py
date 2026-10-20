from typing import Literal

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry, warn_no_image_support


@encoder_registry.register("sd3_clip")
class Sd3ClipEncoder(BaseEncoder[CLIPTextModelWithProjection]):
    """CLIP text encoder for SD3-family models.

    Unlike :class:`ClipTextEncoder` (a plain ``CLIPTextModel`` returning only
    ``pooler_output``), SD3 needs **both** the penultimate hidden states (used as
    part of the sequence conditioning) and the **projected** pooled embedding
    (``text_embeds``) from a ``CLIPTextModelWithProjection``.
    :meth:`encode_seq_pooled` returns both; :meth:`encode` returns just the sequence
    part to satisfy the single-tensor ``Encoder`` interface.
    """

    type: Literal["sd3_clip"] = "sd3_clip"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "CLIPTextModelWithProjection"
    pretrained_model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[CLIPTokenizer] = HfModelLoader(
        library="transformers",
        class_name="CLIPTokenizer",
        pretrained_model_id="stabilityai/stable-diffusion-3.5-medium",
        subfolder="tokenizer",
    )

    max_length: int = 77
    hidden_state_layer: int = -2

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode_seq_pooled(
        self, prompt, images=None, system_prompt: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        outputs = model(
            clip_inputs.input_ids.to(model.device), output_hidden_states=True
        )
        seq = outputs.hidden_states[self.hidden_state_layer]
        pooled = outputs.text_embeds
        return seq, pooled

    def encode(self, prompt, images=None, system_prompt: str | None = None):
        return self.encode_seq_pooled(prompt, images, system_prompt)[0]
