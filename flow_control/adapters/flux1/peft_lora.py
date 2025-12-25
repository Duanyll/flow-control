from typing import Literal, cast

from diffusers import FluxControlPipeline

from flow_control.utils.logging import get_logger
from flow_control.utils.upcasting import cast_trainable_parameters

from ..peft_lora import PeftLoraAdapter
from .base import BaseFlux1Adapter

logger = get_logger(__name__)
NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


class Flux1PeftLoraAdapter(PeftLoraAdapter, BaseFlux1Adapter):
    """
    Adapter for LoRA fine-tuning using the PEFT library.
    """

    pretrained_lora_id: str | None = None

    train_norm_layers: bool = False
    lora_layers: Literal["all-linear"] | list[str] = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    rank: int = 128
    gaussian_init_lora: bool = False
    use_lora_bias: bool = False

    def load_transformer(self):
        super().load_transformer()

        if self.pretrained_lora_id is not None:
            lora_state_dict = cast(
                dict, FluxControlPipeline.lora_state_dict(self.pretrained_lora_id)
            )
            self.load_model(lora_state_dict)
            cast_trainable_parameters(self.transformer, self.trainable_dtype)

    def _install_modules(self):
        transformer = self.transformer

        if self.train_norm_layers:
            for name, param in transformer.named_parameters():
                if any(k in name for k in NORM_LAYER_PREFIXES):
                    param.requires_grad = True

        super()._install_modules()
