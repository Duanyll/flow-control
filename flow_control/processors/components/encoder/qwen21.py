from typing import Literal

import torch
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.tensor import tensor_to_pil
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, encoder_registry


@encoder_registry.register("qwen21")
class QwenImage21Encoder(BaseEncoder[Qwen3VLForConditionalGeneration]):
    type: Literal["qwen21"] = "qwen21"
    library: Literal["transformers"] = "transformers"
    class_name: str = "Qwen3VLForConditionalGeneration"
    pretrained_model_id: str = "Qwen/Qwen-Image-2.1"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16
    processor: HfModelLoader[Qwen3VLProcessor] = HfModelLoader(
        library="transformers",
        class_name="Qwen3VLProcessor",
        pretrained_model_id="Qwen/Qwen-Image-2.1",
        subfolder="processor",
        # Transformers 5.17 AutoTokenizer otherwise requires config.json in
        # this tokenizer-only subfolder before reading tokenizer_config.json.
        extra_from_pretrained_kwargs={"tokenizer_type": "qwen2"},
    )

    def load_model(self, device: torch.device, frozen: bool = True) -> bool:
        self.processor.load_model(device)
        return super().load_model(device, frozen)

    def encode_condition(
        self, prompt: str, images: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # Reuse the checkpoint's raw template, image-slot masks, and pre-final-
        # RMSNorm extraction. This lightweight pipeline loads no extra models.
        pipe = QwenImage21Pipeline(
            text_encoder=self.model,
            processor=self.processor.model,
            # Diffusers accepts absent components at runtime; annotations omit None.
            vae=None,  # ty: ignore[invalid-argument-type]
            transformer=None,  # ty: ignore[invalid-argument-type]
            scheduler=None,
        )
        return pipe.encode_prompt(
            prompt,
            image=[tensor_to_pil(img) for img in images] if images else None,
            device=self.model.device,
        )
