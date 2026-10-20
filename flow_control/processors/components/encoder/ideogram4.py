from typing import Any, Literal

import torch
from einops import rearrange
from transformers import Qwen2Tokenizer, Qwen3VLTextModel
from transformers.masking_utils import create_causal_mask

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry, warn_no_image_support

IDEOGRAM4_REPO = "ideogram-ai/ideogram-4-fp8"


@encoder_registry.register("ideogram4")
class Ideogram4Encoder(BaseEncoder[Qwen3VLTextModel]):
    """Ideogram 4 text features: 13 tapped Qwen3-VL-8B decoder layers, ``(hidden, layer)``
    interleaved into ``[1, L, 13 * 4096]``.

    Replicates the official ``_encode_text``: the bare chat template (no system prompt),
    the decoder layers driven by hand so layer 35 is tapped before the final norm. The
    official encoder also sees the packed image slots after the text, masked out; the
    encoder is causal, so the text features are the same without them. The repo stores
    the encoder weight-only fp8; only the language model is built (the vision tower is
    unused).
    """

    type: Literal["ideogram4"] = "ideogram4"
    library: Literal["transformers"] = "transformers"
    class_name: str = "Qwen3VLTextModel"
    pretrained_model_id: str = IDEOGRAM4_REPO
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[Qwen2Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id=IDEOGRAM4_REPO,
        subfolder="tokenizer",
    )

    max_sequence_length: int = 2048
    select_layers: list[int] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]

    def _convert(self, key: str, tensor: torch.Tensor):
        if key.startswith("language_model."):
            yield key.removeprefix("language_model."), tensor

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt=None):
        # apply_chat_template is typed for string contents only.
        tokenizer: Any = self.tokenizer.model
        model = self.model
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        assert isinstance(text, str)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(model.device)
        if ids.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"The prompt has {ids.shape[1]} tokens, above "
                f"max_sequence_length={self.max_sequence_length}."
            )
        embeds = model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        mask = create_causal_mask(
            config=model.config,
            inputs_embeds=embeds,
            attention_mask=torch.ones_like(ids),
            past_key_values=None,
            position_ids=positions,
        )
        rope = model.rotary_emb(embeds, positions[None].expand(3, -1, -1))
        taps = []
        hidden = embeds
        for index, layer in enumerate(model.layers):
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                position_embeddings=rope,
            )
            if index in self.select_layers:
                taps.append(hidden)
        return rearrange(torch.stack(taps, dim=-1), "b l h k -> b l (h k)")
