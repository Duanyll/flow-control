from typing import Literal, cast

from diffusers import QwenImagePipeline

from flow_control.utils.logging import get_logger
from flow_control.utils.upcasting import cast_trainable_parameters

from ..peft_lora import PeftLoraAdapter
from .base import BaseQwenImageAdapter

logger = get_logger(__name__)


class QwenPeftLoraAdapter(PeftLoraAdapter, BaseQwenImageAdapter):
    """
    Adapter for LoRA fine-tuning using the PEFT library.
    """

    pretrained_lora_id: str | None = None

    lora_layers: Literal["all-linear"] | list[str] = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "img_mlp.net.2",
        "img_mod.1",
        "txt_mlp.net.2",
        "txt_mod.1",
    ]
    exclude_lora_layers: list[str] = [
        # Exclude the last layer's text layers
        "59.txt_mlp.net.2",
        "59.attn.to_add_out",
    ]
    rank: int = 16
    gaussian_init_lora: bool = False
    use_lora_bias: bool = False

    def load_transformer(self):
        super().load_transformer()

        if self.pretrained_lora_id is not None:
            lora_state_dict = cast(
                dict, QwenImagePipeline.lora_state_dict(self.pretrained_lora_id)
            )
            self.load_model(lora_state_dict)
            cast_trainable_parameters(self.transformer, self.trainable_dtype)
