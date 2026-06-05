from typing import Any, Literal

import torch
import torchvision.transforms as T
from pydantic import ConfigDict, PrivateAttr

from flow_control.utils import device as devutil

from .base import BaseReward


class CLIPImageSimilarityReward(BaseReward):
    """CLIP-based image-to-image cosine similarity reward.

    Encodes ``batch["clean_image"]`` and ``batch["reference_image"]`` with a
    CLIP image encoder, L2-normalizes the embeddings, and returns their cosine
    similarity in ``[-1, 1]`` (typically in ``[0, 1]`` for natural images).

    The raw cosine similarity is returned as-is; configure the optional
    :attr:`normalize` transform on the base class if you want to remap the
    range (e.g. clamp to ``[0, 1]``).
    """

    type: Literal["clip_image_similarity"] = "clip_image_similarity"
    model_name: str = "openai/clip-vit-large-patch14"

    model_config = ConfigDict(extra="forbid")

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _transform: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "reference_image"}

    def _load_model(self, device: torch.device) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._device = device
        model = CLIPModel.from_pretrained(self.model_name)
        model.eval()
        # pyright incorrectly resolves CLIPModel.to() due to transformers stubs
        self._model = model.to(device)  # type: ignore[reportArgumentType]
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
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute CLIP image-similarity score for a single sample.

        Expects ``batch["clean_image"]`` and ``batch["reference_image"]``, both
        ``[1, C, H, W]`` tensors in ``[0, 1]``.
        """
        clean = batch["clean_image"]
        reference = batch["reference_image"]

        clean_embeds = self._encode(clean)
        reference_embeds = self._encode(reference)

        # Cosine similarity via dot product of L2-normalized embeddings.
        sim = (clean_embeds * reference_embeds).sum(dim=-1)  # [1]
        return sim

    def _unload_model(self) -> None:
        import gc

        del self._model, self._processor, self._transform
        self._model = None
        self._processor = None
        self._transform = None
        gc.collect()
        devutil.empty_cache()


if __name__ == "__main__":
    import urllib.request
    from pathlib import Path

    from PIL import Image, ImageOps
    from rich import print

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

    reward = CLIPImageSimilarityReward()
    reward.load_model(devutil.default_device())

    same_batch = {
        "clean_image": image_tensor,
        "reference_image": image_tensor,
    }
    same_score = reward.score(same_batch)
    print(f"[bold]Identical image similarity:[/] {same_score.aggregate().item():.4f}")
    assert same_score.aggregate().item() >= 0.99, (
        f"Expected identical similarity >= 0.99, got {same_score.aggregate().item()}"
    )

    inverted_batch = {
        "clean_image": image_tensor,
        "reference_image": inverted_tensor,
    }
    inverted_score = reward.score(inverted_batch)
    print(
        f"[bold]Inverted image similarity:[/] {inverted_score.aggregate().item():.4f}"
    )
    assert inverted_score.aggregate().item() < same_score.aggregate().item(), (
        f"Expected inverted similarity < identical similarity, got "
        f"{inverted_score.aggregate().item()} vs {same_score.aggregate().item()}"
    )

    reward.unload_model()
    print("[bold green]Self-test passed.[/]")
