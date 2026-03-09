from typing import Any, Literal

import torch
from pydantic import ConfigDict, PrivateAttr

from .base import BaseReward


class PickScoreReward(BaseReward):
    """PickScore-based preference reward."""

    type: Literal["pickscore"] = "pickscore"
    processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_name: str = "yuvalkirstain/PickScore_v1"

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    def load_model(self, device: torch.device) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._device = device
        self._processor = CLIPProcessor.from_pretrained(self.processor_name)
        model = CLIPModel.from_pretrained(self.model_name)
        model.eval()
        # pyright incorrectly resolves CLIPModel.to() due to transformers stubs
        self._model = model.to(device)  # type: ignore[reportArgumentType]

    @torch.no_grad()
    def score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute PickScore for a single sample.

        Expects ``batch["clean_image"]`` ([1, C, H, W] in [0, 1]) and
        ``batch["prompt"]`` (str).
        """
        from flow_control.utils.common import tensor_to_pil

        image = batch["clean_image"]  # [1, C, H, W]
        prompt = batch["prompt"]

        # PickScore processor expects PIL images
        pil_image = tensor_to_pil(image)

        image_inputs = self._processor(
            images=[pil_image],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self._device) for k, v in image_inputs.items()}

        text_inputs = self._processor(
            text=[prompt] if isinstance(prompt, str) else prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self._device) for k, v in text_inputs.items()}

        image_embs = self._model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self._model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self._model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # Normalize to ~[0, 1] range
        return scores / 26.0

    def unload_model(self) -> None:
        import gc

        del self._model, self._processor
        self._model = None
        self._processor = None
        gc.collect()
        torch.cuda.empty_cache()
