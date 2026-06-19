"""HPSv2.1 reward: Human Preference Score v2.1.

Computes the HPSv2.1 score (paper-range ~[0.20, 0.32]) between a generated
image and its text prompt.  The model is the official HPSv2.1 fine-tune
of ``open_clip``'s ``ViT-H-14`` (``laion2b_s32b_b79k``).

Implementation notes:
- Uses ``open_clip.create_model_and_transforms`` to build the base CLIP
  (open_clip will auto-download the laion2b weights into the HF cache on
  first use).
- HPSv2.1 fine-tuning weights are loaded from ``HPS_v2.1_compressed.pt``
  (key ``"state_dict"``).  When ``checkpoint_path`` is None, the file is
  auto-downloaded via ``huggingface_hub.hf_hub_download`` from
  ``xswu/HPSv2``, which respects the standard ``HF_HUB_CACHE`` /
  ``HF_ENDPOINT`` / ``HF_HUB_OFFLINE`` env vars.
- The image preprocessing replicates the official HPSv2 ``ResizeMaxSize``
  exactly, including the documented ``fn`` quirk (see below) so scores
  match the published numbers.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import ConfigDict, PrivateAttr
from torchvision.transforms import Compose, InterpolationMode, Normalize
from torchvision.transforms import functional as TF

from flow_control.utils import device as devutil
from flow_control.utils.logging import get_logger
from flow_control.utils.types import TorchDType

from .base import BaseReward, reward_registry

logger = get_logger(__name__)

HPSV2_HF_REPO = "xswu/HPSv2"
HPSV2_CKPT_FILENAME = "HPS_v2.1_compressed.pt"


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class ResizeMaxSize(nn.Module):
    """Resize the longer side to ``max_size`` and pad the shorter side.

    Ported verbatim from the official HPSv2 preprocessing.

    Quirk (intentionally preserved for HPSv2.1 score parity): the original
    code reads ``self.fn = min if fn == "min" else min``, so both branches
    return ``min``.  This is a known bug in the upstream HPSv2 codebase,
    but it bakes into the published HPSv2.1 numbers — replicating it
    byte-for-byte is required for our scores to match.
    """

    def __init__(
        self,
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        fn: str = "max",
        fill: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        # NOTE: do NOT "fix" this — must match upstream HPSv2.1 byte-for-byte.
        # Both 'min' and 'max' map to min.  See class docstring.
        self.fn = min if fn == "min" else min
        self.fill = fill

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if isinstance(img, torch.Tensor):
            # Assuming NCHW or CHW; take H, W from the last two dims.
            height, width = img.shape[-2:]
        else:
            width, height = img.size  # type: ignore[attr-defined]
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = TF.resize(img, list(new_size), self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = TF.pad(
                img,
                padding=[
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - pad_w // 2,
                    pad_h - pad_h // 2,
                ],
                fill=self.fill,
            )
        return img


class MaskAwareNormalize(nn.Module):
    """Normalize that leaves an optional alpha channel untouched.

    For NCHW inputs with C == 4, the first 3 channels are normalized and
    the alpha channel is preserved.  Otherwise, behaves like the standard
    :class:`torchvision.transforms.Normalize`.
    """

    def __init__(
        self,
        mean: tuple[float, ...] | list[float],
        std: tuple[float, ...] | list[float],
    ) -> None:
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4 and tensor.shape[1] == 4:
            normalized_parts: list[torch.Tensor] = []
            for i in range(tensor.shape[0]):
                img_slice = tensor[i]
                normalized_rgb = self.normalize(img_slice[:3])
                alpha_channel = img_slice[3:]
                normalized_parts.append(
                    torch.cat([normalized_rgb, alpha_channel], dim=0)
                )
            return torch.stack(normalized_parts, dim=0)
        return self.normalize(tensor)


def image_transform_tensor(
    image_size: int,
    mean: tuple[float, ...] | list[float] | None = None,
    std: tuple[float, ...] | list[float] | None = None,
    fill_color: int = 0,
) -> Compose:
    """Build the HPSv2.1 preprocessing pipeline for tensor inputs."""
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD

    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    normalize = MaskAwareNormalize(mean=mean, std=std)

    transforms: list[nn.Module] = [
        ResizeMaxSize(image_size, fill=fill_color),
        normalize,
    ]
    return Compose(transforms)


@reward_registry.register("hpsv2")
class HPSv2Reward(BaseReward):
    """Human Preference Score v2.1 reward.

    Expects ``batch["clean_image"]`` ([1, C, H, W] in [0, 1]) and
    ``batch["prompt"]`` (str).  Returns a single-component ``[1]`` tensor
    holding the raw paired dot-product (typically ~0.20-0.35).
    """

    type: Literal["hpsv2"] = "hpsv2"
    model_name: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    checkpoint_path: str | None = None
    """Path to ``HPS_v2.1_compressed.pt``.

    When None, the file is auto-downloaded from the ``xswu/HPSv2`` repo on
    Hugging Face Hub.
    """
    dtype: TorchDType = torch.float32

    model_config = ConfigDict(extra="forbid")

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _transform: Any = PrivateAttr(default=None)
    _device: torch.device | None = PrivateAttr(default=None)

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _resolve_checkpoint_path(self) -> str:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=HPSV2_HF_REPO, filename=HPSV2_CKPT_FILENAME)

    def _load_model(self, device: torch.device) -> None:
        import open_clip

        self._device = device

        logger.info(
            f"Loading open_clip base model {self.model_name} "
            f"(pretrained={self.pretrained})..."
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            precision="amp",
            device=str(device),
            output_dict=True,
        )

        visual: Any = model.visual
        image_mean = getattr(visual, "image_mean", None)
        image_std = getattr(visual, "image_std", None)
        image_size = visual.image_size
        if isinstance(image_size, tuple):
            image_size = image_size[0]

        transform = image_transform_tensor(
            image_size=image_size,
            mean=image_mean,
            std=image_std,
        )

        ckpt_path = self._resolve_checkpoint_path()
        logger.info(f"Loading HPSv2.1 fine-tuning weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"])
        if missing:
            logger.warning(f"HPSv2.1 ckpt missing keys (first 5): {list(missing)[:5]}")
        if unexpected:
            logger.warning(
                f"HPSv2.1 ckpt unexpected keys (first 5): {list(unexpected)[:5]}"
            )

        model.eval()
        # Re-assign to device in case load_state_dict moved any buffers.
        model = model.to(device)

        self._model = model
        self._transform = transform
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        logger.info("HPSv2.1 reward loaded.")

    @torch.no_grad()
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        assert self._model is not None, "HPSv2Reward not loaded; call load_model first."
        assert self._device is not None

        image: torch.Tensor = batch["clean_image"]  # [1, C, H, W] in [0, 1]
        prompt: str = batch["prompt"]

        image = image.to(device=self._device, dtype=self.dtype, non_blocking=True)
        pixels = self._transform(image)

        tokens = self._tokenizer([prompt] if isinstance(prompt, str) else prompt)
        tokens = tokens.to(device=self._device, non_blocking=True)

        outputs = self._model(pixels, tokens)
        image_features = outputs["image_features"]
        text_features = outputs["text_features"]

        # open_clip already L2-normalizes features when output_dict=True.
        logits = image_features @ text_features.T
        hps_score = torch.diagonal(logits, 0).contiguous()
        # Return [1] raw tensor; downstream normalize handles rescaling.
        return hps_score.float().view(1)

    def _unload_model(self) -> None:
        import gc

        del self._model, self._tokenizer, self._transform
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._device = None
        gc.collect()
        devutil.empty_cache()


if __name__ == "__main__":
    import urllib.request
    from pathlib import Path

    from PIL import Image
    from rich import print

    from flow_control.utils.tensor import pil_to_tensor

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = data_dir / "000000039769.jpg"
    if not image_path.exists():
        print(f"[bold]Downloading test image to[/] {image_path} ...")
        urllib.request.urlretrieve(image_url, image_path)
        print("[bold green]Done.[/]")
    else:
        print(f"[bold]Test image already exists at[/] {image_path}")

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image_pil)  # [1, C, H, W] in [0, 1]
    print(
        f"[bold]Image size:[/] {image_pil.size}, tensor shape: {tuple(image_tensor.shape)}"
    )

    device = devutil.default_device()
    print(f"[bold]Device:[/] {device}")

    reward = HPSv2Reward()
    reward.load_model(device)

    matching_prompt = "Two cats lying on a couch with two tv remotes"
    mismatching_prompt = "A picture of a red sports car on a race track"

    batch_match: dict[str, Any] = {
        "clean_image": image_tensor,
        "prompt": matching_prompt,
    }
    score_match = reward.score(batch_match)
    print(
        f"[bold]HPSv2.1[/] (matching prompt: {matching_prompt!r}) "
        f"= [green]{score_match.aggregate().item():.4f}[/]"
    )

    batch_mismatch: dict[str, Any] = {
        "clean_image": image_tensor,
        "prompt": mismatching_prompt,
    }
    score_mismatch = reward.score(batch_mismatch)
    print(
        f"[bold]HPSv2.1[/] (mismatching prompt: {mismatching_prompt!r}) "
        f"= [red]{score_mismatch.aggregate().item():.4f}[/]"
    )

    assert score_match.aggregate().item() > score_mismatch.aggregate().item(), (
        f"Expected matching prompt score ({score_match.aggregate().item():.4f}) > "
        f"mismatching prompt score ({score_mismatch.aggregate().item():.4f})"
    )
    print("[bold green]OK — matching prompt scores higher than mismatching prompt.[/]")

    reward.unload_model()
