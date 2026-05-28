"""Aesthetic reward using the improved-aesthetic-predictor.

Ports the LAION ``improved-aesthetic-predictor``
(https://github.com/christophschuhmann/improved-aesthetic-predictor) into the
``flow_control`` rewards framework.

Algorithm:

1. CLIP ViT-L/14 (``openai/clip-vit-large-patch14``) image encoder produces a
   768-dimensional feature vector.
2. The feature is L2-normalised along the last dimension.
3. A 5-layer MLP regression head (768 -> 1024 -> 128 -> 64 -> 16 -> 1) with
   Dropout(0.2)/Dropout(0.2)/Dropout(0.1) interleaved between the first three
   linear layers (and no activations between layers, matching the original
   checkpoint) maps the embedding to a scalar score.
4. Typical raw scores fall in roughly ``[3, 8]``; users that want
   ``~[0, 1]`` should configure ``normalize=AffineNormalize(scale=0.1)`` on
   their reward config.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pydantic import ConfigDict, PrivateAttr
from transformers import CLIPModel, CLIPProcessor

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import tensor_to_pil

from .base import BaseReward

logger = get_logger(__name__)


class _AestheticMLP(nn.Module):
    """5-layer MLP regression head for the improved aesthetic predictor.

    Matches the architecture saved in ``sac+logos+ava1-l14-linearMSE.pth``:
    ``Linear(768, 1024) -> Dropout(0.2) -> Linear(1024, 128) -> Dropout(0.2)
    -> Linear(128, 64) -> Dropout(0.1) -> Linear(64, 16) -> Linear(16, 1)``.
    No non-linear activations between layers (this is the original
    checkpoint's quirk).
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.layers(embed)


class AestheticReward(BaseReward):
    """LAION improved aesthetic predictor reward.

    Returns the raw aesthetic score (typical range ``~[3, 8]``).  Configure
    ``normalize`` on the parent reward config to rescale into ``[0, 1]``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["aesthetic"] = "aesthetic"

    clip_model: HfModelLoader[CLIPModel] = HfModelLoader(
        library="transformers",
        class_name="CLIPModel",
        pretrained_model_id="openai/clip-vit-large-patch14",
        dtype=torch.float32,
    )

    processor: HfModelLoader[CLIPProcessor] = HfModelLoader(
        library="transformers",
        class_name="CLIPProcessor",
        pretrained_model_id="openai/clip-vit-large-patch14",
        dtype=torch.float32,
    )

    checkpoint_path: str | None = None
    """Local path to the MLP weights ``.pth`` file.

    When ``None`` (the default) the file is downloaded from
    ``camenduru/improved-aesthetic-predictor`` on the Hugging Face Hub.
    """

    _mlp: _AestheticMLP | None = PrivateAttr(default=None)
    _device: torch.device | None = PrivateAttr(default=None)

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image"}

    def _resolve_checkpoint_path(self) -> str:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        logger.info(
            "Downloading improved-aesthetic-predictor checkpoint "
            "(sac+logos+ava1-l14-linearMSE.pth) from camenduru/improved-aesthetic-predictor"
        )
        return hf_hub_download(
            repo_id="camenduru/improved-aesthetic-predictor",
            filename="sac+logos+ava1-l14-linearMSE.pth",
        )

    def _load_model(self, device: torch.device) -> None:
        self._device = device

        self.clip_model.load_model(device=device)
        self.processor.load_model(device=device)

        mlp = _AestheticMLP()
        ckpt_path = self._resolve_checkpoint_path()
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        mlp.to(device)
        self._mlp = mlp
        logger.info(f"Loaded AestheticReward MLP head from {ckpt_path} on {device}")

    @torch.no_grad()
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute raw aesthetic score for a single sample.

        Expects ``batch["clean_image"]`` of shape ``[1, C, H, W]`` in ``[0, 1]``.
        Returns a ``[1]`` tensor with the raw scalar score (NOT normalised --
        ``BaseReward.score`` applies the configured normalize transform).
        """
        if self._mlp is None or self._device is None:
            raise RuntimeError(
                "AestheticReward is not loaded; call load_model(device) first."
            )

        image: torch.Tensor = batch["clean_image"]  # [1, C, H, W] in [0, 1]

        pil_image = tensor_to_pil(image)
        processor: Any = self.processor.model
        clip: Any = self.clip_model.model

        inputs = processor(images=[pil_image], return_tensors="pt")
        inputs = {
            k: v.to(device=self._device, dtype=clip.dtype) for k, v in inputs.items()
        }

        embed = clip.get_image_features(**inputs).pooler_output
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        score = self._mlp(embed).squeeze(-1)  # [1]
        return score

    def _unload_model(self) -> None:
        import gc

        if self._mlp is not None:
            del self._mlp
            self._mlp = None
        self.clip_model.unload_model()
        self.processor.unload_model()
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import urllib.request
    from pathlib import Path

    import numpy as np
    from PIL import Image
    from rich import print as rprint

    from flow_control.utils.tensor import pil_to_tensor

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = data_dir / "000000039769.jpg"
    if not image_path.exists():
        rprint(f"[bold]Downloading test image to[/] {image_path} ...")
        urllib.request.urlretrieve(image_url, image_path)
        rprint("[bold green]Done.[/]")

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image_pil)  # [1, C, H, W] in [0, 1]
    rprint(
        f"[bold]COCO image size:[/] {image_pil.size}, tensor shape: {image_tensor.shape}"
    )

    # Build a deliberately low-quality 64x64 noise image as a contrast sample.
    rng = np.random.default_rng(0)
    noise_array = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    noise_pil = Image.fromarray(noise_array)
    noise_tensor = pil_to_tensor(noise_pil)
    rprint(
        f"[bold]Noise image size:[/] {noise_pil.size}, tensor shape: {noise_tensor.shape}"
    )

    reward = AestheticReward()
    reward.load_model(torch.device("cuda"))

    coco_score = reward.score({"clean_image": image_tensor})
    noise_score = reward.score({"clean_image": noise_tensor})

    rprint(f"[bold]COCO aesthetic score:[/]  {coco_score.item():.4f}")
    rprint(f"[bold]Noise aesthetic score:[/] {noise_score.item():.4f}")

    assert coco_score.item() > noise_score.item(), (
        f"Expected COCO image to score higher than random noise: "
        f"coco={coco_score.item():.4f} noise={noise_score.item():.4f}"
    )
    rprint("[bold green]COCO image scored higher than noise, as expected.[/]")

    reward.unload_model()
