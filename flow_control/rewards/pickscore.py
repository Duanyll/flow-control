from typing import Any, Literal

import torch
from pydantic import ConfigDict
from transformers import CLIPModel, CLIPProcessor

from flow_control.utils.hf_model import HfModelLoader

from .base import BaseReward


class PickScoreReward(BaseReward):
    """PickScore-based preference reward."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pickscore"] = "pickscore"

    processor: HfModelLoader[CLIPProcessor] = HfModelLoader(
        library="transformers",
        class_name="CLIPProcessor",
        pretrained_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        dtype=torch.float32,
    )

    model: HfModelLoader[CLIPModel] = HfModelLoader(
        library="transformers",
        class_name="CLIPModel",
        pretrained_model_id="yuvalkirstain/PickScore_v1",
        dtype=torch.float32,
    )

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _load_model(self, device: torch.device) -> None:
        self._device = device
        self.processor.load_model(device=device)
        self.model.load_model(device=device)

    @torch.no_grad()
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute PickScore for a single sample.

        Expects ``batch["clean_image"]`` ([1, C, H, W] in [0, 1]) and
        ``batch["prompt"]`` (str).
        """
        from flow_control.utils.tensor import tensor_to_pil

        image = batch["clean_image"]  # [1, C, H, W]
        prompt = batch["prompt"]

        # PickScore processor expects PIL images
        pil_image = tensor_to_pil(image)
        processor: Any = self.processor.model
        model: Any = self.model.model

        image_inputs = processor(
            images=[pil_image],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = image_inputs.to(device=self._device, dtype=model.dtype)

        text_inputs = processor(
            text=[prompt] if isinstance(prompt, str) else prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = text_inputs.to(device=self._device, dtype=model.dtype)

        image_embs = model.get_image_features(**image_inputs).pooler_output
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs).pooler_output
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # Normalize to ~[0, 1] range
        return scores / 26.0


if __name__ == "__main__":
    import urllib.request
    from pathlib import Path

    from PIL import Image
    from rich import print as rprint

    from flow_control.utils.tensor import pil_to_tensor

    data_dir = Path("data")
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = data_dir / "000000039769.jpg"
    if not image_path.exists():
        rprint(f"[bold]Downloading test image to[/] {image_path} ...")
        urllib.request.urlretrieve(image_url, image_path)
        rprint("[bold green]Done.[/]")

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image_pil)  # [1, C, H, W] in [0, 1]
    rprint(f"[bold]Image size:[/] {image_pil.size}, tensor shape: {image_tensor.shape}")

    # Quick test
    reward = PickScoreReward()
    reward.load_model(torch.device("cuda:2"))

    batch = {
        "clean_image": image_tensor,
        "prompt": "A dog playing with a ball in the park.",
    }
    score = reward.score(batch)
    rprint(f"[bold]PickScore reward:[/] {score.item():.4f}")  # should be low

    batch = {
        "clean_image": image_tensor,
        "prompt": "Two cats lying on a couch with two tv remotes.",
    }
    score = reward.score(batch)
    rprint(
        f"[bold]PickScore reward:[/] {score.item():.4f}"
    )  # should be higher than the previous one
