from typing import Any, Literal

import torch
import torchvision.transforms as T
from pydantic import ConfigDict, PrivateAttr
from torch import Tensor

from flow_control.samplers.node import Row
from flow_control.utils.condition_image import ConditionImage, ConditionImageSpec

from .base import BaseScorer, LocalReward, scorer_registry


@scorer_registry.register("clip_image_similarity")
class ClipImageSimilarityScorer(BaseScorer, LocalReward[Row, Tensor]):
    """CLIP-based image-to-image cosine similarity reward.

    Encodes ``row["clean_image"]`` and the :attr:`reference` condition image
    with a CLIP image encoder, L2-normalizes the embeddings, and returns their cosine
    similarity in ``[-1, 1]`` (typically in ``[0, 1]`` for natural images).

    The raw cosine similarity is returned as-is; configure :attr:`normalize`
    to remap the range (e.g. clamp to ``[0, 1]``).
    """

    type: Literal["clip_image_similarity"] = "clip_image_similarity"
    model_name: str = "openai/clip-vit-large-patch14"
    reference: ConditionImageSpec = "reference"
    """Condition image to compare against, e.g. ``"reference[1]"`` or ``"control"``."""

    model_config = ConfigDict(extra="forbid")

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _transform: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    @property
    def fields(self) -> set[str]:
        return {"clean_image", ConditionImage.parse(self.reference).image_field}

    def load(self, device: torch.device) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._device = device
        model = CLIPModel.from_pretrained(self.model_name)
        model.eval()
        # transformers wraps `to()` in functools.wraps, so both checkers read the
        # decorated signature as unbound and take `device` for `self`.
        self._model = model.to(device)  # ty: ignore[invalid-argument-type]
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

        # Build image transform from processor config (same logic as
        # ``clip_score.py``).
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

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        """Run a ``[1, C, H, W]`` image in ``[0, 1]`` through CLIP image encoder.

        Returns the L2-normalized image embedding of shape ``[1, D]``.
        """
        pixels = self._transform(image).to(device=self._device, dtype=self._model.dtype)
        embeds = self._model.get_image_features(pixel_values=pixels)
        # Some transformers versions return a BaseModelOutputWithPooling
        # instead of a tensor; unwrap if so.
        if hasattr(embeds, "pooler_output"):
            embeds = embeds.pooler_output
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def run_batch(self, rows: list[Row]) -> list[Tensor]:
        return [self._score_row(row) for row in rows]

    def _score_row(self, row: Row) -> Tensor:
        """Compute CLIP image-similarity score for a single sample.

        Expects ``row["clean_image"]`` and the :attr:`reference` image, both
        ``[1, C, H, W]`` tensors in ``[0, 1]``.
        """
        clean = row["clean_image"]
        reference = ConditionImage.parse(self.reference).image(row)

        clean_embeds = self._encode(clean)
        reference_embeds = self._encode(reference)

        # Cosine similarity via dot product of L2-normalized embeddings.
        sim = (clean_embeds * reference_embeds).sum(dim=-1)  # [1]
        return sim


if __name__ == "__main__":
    import urllib.request
    from pathlib import Path

    from PIL import Image, ImageOps
    from rich import print

    from flow_control.utils import device as devutil
    from flow_control.utils.tensor import pil_to_tensor

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = data_dir / "000000039769.jpg"
    if not image_path.exists():
        print(f"[bold]Downloading test image to[/] {image_path} ...")
        urllib.request.urlretrieve(image_url, image_path)
        print("[bold green]Done.[/]")

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image_pil)
    inverted_pil = ImageOps.invert(image_pil)
    inverted_tensor = pil_to_tensor(inverted_pil)
    print(f"[bold]Image size:[/] {image_pil.size}, tensor shape: {image_tensor.shape}")

    reward = ClipImageSimilarityScorer()
    reward.load(devutil.default_device())

    same, inverted = (
        score.item()
        for score in reward.run_batch(
            [
                {"clean_image": image_tensor, "reference_images": [image_tensor]},
                {"clean_image": image_tensor, "reference_images": [inverted_tensor]},
            ]
        )
    )
    print(f"[bold]Image similarity:[/] identical {same:.4f}, inverted {inverted:.4f}")
    assert same >= 0.99, f"Expected identical similarity >= 0.99, got {same}"
    assert inverted < same, (
        f"Expected inverted similarity < identical similarity, got {inverted} vs {same}"
    )
    print("[bold green]Self-test passed.[/]")
