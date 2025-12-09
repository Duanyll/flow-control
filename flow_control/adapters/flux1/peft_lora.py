from typing import Literal, cast

import torch
from accelerate.utils import extract_model_from_parallel
from diffusers import FluxControlPipeline
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from flow_control.utils.logging import get_logger
from flow_control.utils.upcasting import cast_trainable_parameters

from .base import BaseFlux1Adapter

logger = get_logger(__name__)
NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


class Flux1PeftLoraAdapter(BaseFlux1Adapter):
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

        target_modules = []
        if self.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in self.lora_layers]
        elif self.lora_layers == "all-linear":
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)

        transformer_lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            init_lora_weights="gaussian" if self.gaussian_init_lora else True,
            target_modules=target_modules,
            lora_bias=self.use_lora_bias,
        )

        transformer.add_adapter(transformer_lora_config)

    def unwrap_transformer(self):
        model = extract_model_from_parallel(self.transformer)
        model = model._orig_mod if hasattr(model, "_orig_mod") else model
        return model

    def save_model(self) -> dict:
        transformer = self.unwrap_transformer()

        layers_to_save = get_peft_model_state_dict(transformer)
        for name, param in transformer.named_parameters():
            if "lora" not in name and param.requires_grad:
                layers_to_save[name] = param.data

        return layers_to_save

    def load_model(self, state_dict: dict):
        transformer = self.transformer

        lora_state_dict = {}
        other_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("transformer.", "")
            k = k.replace("module.", "")
            k = k.replace("default.", "")
            if "lora" in k:
                lora_state_dict[k] = v
            else:
                other_state_dict[k] = v
        incompatible_keys = set_peft_model_state_dict(
            transformer, lora_state_dict, adapter_name="default"
        )
        if incompatible_keys:
            logger.warning(
                f"Some keys in the state_dict are incompatible with the model: {incompatible_keys}"
            )
        incompatible_keys = transformer.load_state_dict(other_state_dict, strict=False)
        if incompatible_keys:
            logger.warning(
                f"Some keys in the state_dict are incompatible with the model: {incompatible_keys}"
            )
