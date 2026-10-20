from typing import Any, Literal

import torch
from pydantic import ConfigDict, Field
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

from flow_control.samplers.node import Row
from flow_control.utils import device as devutil
from flow_control.utils.hf_model import HfModelLoader

from .base import BaseScorer, LocalReward, scorer_registry
from .normalize import AffineNormalize, Normalize


@scorer_registry.register("pickscore")
class PickScoreScorer(BaseScorer, LocalReward[Row, Tensor]):
    """PickScore-based preference reward."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pickscore"] = "pickscore"
    normalize: Normalize = Field(
        default_factory=lambda: AffineNormalize(scale=1.0 / 26.0)
    )

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
    def fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def load(self, device: torch.device) -> None:
        self._device = device
        self.processor.load_model(device=device)
        self.model.load_model(device=device)

    @torch.no_grad()
    def run_batch(self, rows: list[Row]) -> list[Tensor]:
        return [self._score_row(row) for row in rows]

    def _score_row(self, row: Row) -> Tensor:
        """Compute PickScore for a single sample.

        Expects ``row["clean_image"]`` ([1, C, H, W] in [0, 1]) and
        ``row["prompt"]`` (str).
        """
        from flow_control.utils.tensor import tensor_to_pil

        image = row["clean_image"]  # [1, C, H, W]
        prompt = row["prompt"]

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
        # Raw PickScore logits ``[1]``; the configured normalize is applied later.
        return scores.diag()


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
    reward = PickScoreScorer()
    reward.load(devutil.default_device())

    low, high = reward.run_batch(
        [
            {
                "clean_image": image_tensor,
                "prompt": "A dog playing with a ball in the park.",
            },
            {
                "clean_image": image_tensor,
                "prompt": "Two cats lying on a couch with two tv remotes.",
            },
        ]
    )
    rprint(f"[bold]PickScore (raw):[/] dog {low.item():.4f}, cats {high.item():.4f}")
    assert high.item() > low.item(), (low, high)
