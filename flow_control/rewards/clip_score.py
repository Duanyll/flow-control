from typing import Any, Literal

import torch
import torchvision.transforms as T
from pydantic import ConfigDict, PrivateAttr

from .base import BaseReward


class CLIPScoreReward(BaseReward):
    """CLIP-based text-image alignment reward."""

    type: Literal["clip_score"] = "clip_score"
    model_name: str = "openai/clip-vit-large-patch14"

    model_config = ConfigDict(extra="forbid")

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _transform: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _load_model(self, device: torch.device) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._device = device
        model = CLIPModel.from_pretrained(self.model_name)
        model.eval()
        # pyright incorrectly resolves CLIPModel.to() due to transformers stubs
        self._model = model.to(device)  # type: ignore[reportArgumentType]
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

        # Build image transform from processor config
        img_proc = self._processor.image_processor
        config = img_proc.to_dict()
        transforms: list[torch.nn.Module] = []

        if config.get("do_resize"):
            size = config.get("size", {})
            if isinstance(size, int):
                transforms.append(T.Resize(size))
            elif "shortest_edge" in size:
                transforms.append(T.Resize(size["shortest_edge"]))
            elif "height" in size and "width" in size:
                transforms.append(T.Resize((size["height"], size["width"])))

        if config.get("do_center_crop"):
            crop_size = config.get("crop_size", {})
            if isinstance(crop_size, int):
                transforms.append(T.CenterCrop(crop_size))
            elif "height" in crop_size and "width" in crop_size:
                transforms.append(
                    T.CenterCrop((crop_size["height"], crop_size["width"]))
                )

        if config.get("do_normalize"):
            transforms.append(
                T.Normalize(mean=img_proc.image_mean, std=img_proc.image_std)
            )

        self._transform = T.Compose(transforms)

    @torch.no_grad()
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute CLIP score for a single sample.

        Expects ``batch["clean_image"]`` ([1, C, H, W] in [0, 1]) and
        ``batch["prompt"]`` (str).
        """
        image = batch["clean_image"]  # [1, C, H, W]
        prompt = batch["prompt"]

        pixels = self._transform(image).to(device=self._device, dtype=self._model.dtype)
        text_inputs = self._processor(
            text=[prompt] if isinstance(prompt, str) else prompt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self._device) for k, v in text_inputs.items()}
        outputs = self._model(pixel_values=pixels, **text_inputs)
        # Normalize score to ~[0, 1] range
        return outputs.logits_per_image.diagonal() / 30.0

    def _unload_model(self) -> None:
        import gc

        del self._model, self._processor, self._transform
        self._model = None
        self._processor = None
        self._transform = None
        gc.collect()
        torch.cuda.empty_cache()
