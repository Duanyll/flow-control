"""OCR-based reward.

Parses a target string from the prompt (text inside the first pair of double
quotes), runs OCR on the generated image, and scores the match between the
target and the OCR output as ``1 - levenshtein / len(target)`` clipped to
``[0, 1]``.

The OCR backend is built on top of `rapidocr` 3.x (PP-OCR ONNX wrapper, no
PaddlePaddle dependency).  See ``RapidOcrBackend`` for the GPU-vs-CPU
trade-off currently in effect.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import torch
from PIL import Image
from pydantic import ConfigDict, PrivateAttr

from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import tensor_to_pil

from .base import BaseReward

logger = get_logger(__name__)


def _levenshtein(a: str, b: str) -> int:
    """Iterative two-row dynamic programming Levenshtein distance.

    Works in ``O(len(a) * len(b))`` time and ``O(min(len(a), len(b)))``
    memory.  Pure-Python so we do not need an external dependency such as
    ``python-Levenshtein`` or ``rapidfuzz``.
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Make ``b`` the shorter string so the working row is small.
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    current = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        current[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous, current = current, previous
    return previous[len(b)]


class OcrBackend(Protocol):
    """Minimal OCR backend interface used by :class:`OcrReward`."""

    def recognize(self, image: Image.Image) -> list[str]:
        """Return the list of recognized text strings for ``image``."""
        ...


class RapidOcrBackend:
    """OCR backend backed by ``rapidocr`` 3.x.

    GPU integration notes
    ---------------------
    ``rapidocr`` 3.x exposes the ONNX Runtime CUDA execution provider, but
    its ``EngineConfig`` schema does not expose the
    ``gpu_external_alloc`` provider option that would let ONNX Runtime
    share PyTorch's CUDA allocator (see
    ``rapidocr/inference_engine/onnxruntime/provider_config.py``).  The
    only escape hatches are the documented ``cuda_ep_cfg`` keys
    (``device_id``, ``arena_extend_strategy``,
    ``cudnn_conv_algo_search``, ``do_copy_in_default_stream``), all of
    which cause ONNX Runtime to allocate its own CUDA context separate
    from PyTorch's.  That contends for GPU memory with the diffusion
    model and defeats the user's intent of running OCR alongside the
    trainer on the same device.

    Wiring up ``gpu_external_alloc`` against PyTorch's
    :class:`torch.cuda.memory.CUDAPluggableAllocator` would require
    constructing an ``OrtCudaExternalAllocatorInfo`` C struct, encoding
    its address as a string, and managing its lifetime by hand — this is
    too fragile to do here without first-party support from either
    project.

    Therefore this backend currently runs the OCR pipeline on the **CPU
    execution provider**.  We emit a one-time warning so users are aware
    that the ``device`` argument is intentionally ignored.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        from rapidocr import EngineType, LangDet, LangRec, RapidOCR

        # The device argument is accepted for API parity with other rewards
        # but is currently ignored — see class docstring for details.
        if device is not None and device.type == "cuda":
            logger.warning(
                "Falling back to CPU OCR — could not wire up gpu_external_alloc "
                "against PyTorch CUDA allocator; this avoids contending for GPU "
                "memory with the diffusion model."
            )

        params: dict[str, Any] = {
            # Force ONNX Runtime + CPU EP for every task (det / cls / rec).
            # rapidocr 3.x requires Enum instances for these keys.
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.EN,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.EN,
            "EngineConfig.onnxruntime.use_cuda": False,
            # Pin ONNX Runtime to a small thread budget — by default it spawns
            # one intra-op thread per logical CPU and then tries to set
            # affinities, which fails noisily inside Slurm cpusets and
            # contends with the trainer's data loader workers.
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            # Keep rapidocr quiet during training.
            "Global.log_level": "error",
        }
        self._engine = RapidOCR(params=params)

    def recognize(self, image: Image.Image) -> list[str]:
        import numpy as np

        # rapidocr accepts numpy arrays (HWC, uint8, RGB) directly.
        img_array = np.array(image.convert("RGB"))
        result = self._engine(img_array)

        txts = getattr(result, "txts", None)
        if txts is None:
            return []
        return list(txts)


class OcrReward(BaseReward):
    """Reward based on OCR alignment with a target string in the prompt.

    The target is parsed from the prompt as the substring between the first
    pair of double quotes (``prompt.split('"')[1]``).  The image is OCR'd,
    all recognized text is concatenated, and the reward is
    ``1 - levenshtein(target, recognized) / len(target)`` clipped to
    ``[0, 1]``.  If the lowercased + space-stripped target appears verbatim
    in the lowercased + space-stripped OCR output, the distance is treated
    as 0 (perfect score).
    """

    type: Literal["ocr"] = "ocr"

    model_config = ConfigDict(extra="forbid")

    _backend: OcrBackend | None = PrivateAttr(default=None)
    _device: torch.device | None = PrivateAttr(default=None)

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _load_model(self, device: torch.device) -> None:
        self._device = device
        self._backend = RapidOcrBackend(device)

    @torch.no_grad()
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError(
                "OcrReward backend is not loaded; call load_model() first."
            )

        image: torch.Tensor = batch["clean_image"]  # [1, C, H, W] in [0, 1]
        prompt: str = batch["prompt"]
        device = image.device
        dtype = image.dtype

        parts = prompt.split('"')
        if len(parts) < 3:
            logger.warning(
                "Prompt does not contain a double-quoted target; returning 0. "
                "Prompt: %r",
                prompt,
            )
            return torch.zeros(1, device=device, dtype=dtype)

        target = parts[1]
        target_norm = target.replace(" ", "").lower()
        if not target_norm:
            logger.warning(
                "Empty target string parsed from prompt; returning 0. Prompt: %r",
                prompt,
            )
            return torch.zeros(1, device=device, dtype=dtype)

        pil_image = tensor_to_pil(image[0])

        try:
            recognized_list = self._backend.recognize(pil_image)
        except Exception:
            logger.exception("OCR backend failed; assigning max penalty.")
            return torch.zeros(1, device=device, dtype=dtype)

        recognized_concat = "".join(recognized_list).replace(" ", "").lower()

        if target_norm in recognized_concat:
            dist = 0
        else:
            dist = _levenshtein(target_norm, recognized_concat)

        # Cap distance: many unrelated characters should only cost a full
        # mismatch, not blow the reward below zero.
        dist = min(dist, len(target_norm))

        reward = 1.0 - dist / len(target_norm)
        return torch.tensor([reward], device=device, dtype=dtype)

    def _unload_model(self) -> None:
        import gc

        del self._backend
        self._backend = None
        self._device = None
        gc.collect()


if __name__ == "__main__":
    from pathlib import Path

    from PIL import ImageDraw, ImageFont
    from rich import print

    from flow_control.utils.tensor import pil_to_tensor

    test_image_path = Path("./data/ocr_test_hello.png")
    test_image_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a 512x512 white image with "Hello World" in large black text.
    canvas = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(canvas)
    text = "Hello World"

    font: Any = None
    candidate_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for fp in candidate_font_paths:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, size=72)
            break
    if font is None:
        font = ImageFont.load_default()

    # Center the text.
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (512 - tw) // 2 - bbox[0]
    y = (512 - th) // 2 - bbox[1]
    draw.text((x, y), text, fill="black", font=font)
    canvas.save(test_image_path)
    print(f"Saved test image to: {test_image_path}")

    image_tensor = pil_to_tensor(canvas)

    reward_module = OcrReward()
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading OcrReward on device: {test_device}")
    reward_module.load_model(test_device)
    image_tensor = image_tensor.to(test_device)

    positive_prompt = (
        'New York Skyline with "Hello World" written with fireworks on the sky'
    )
    negative_prompt = 'A picture of "Goodbye Mars" on a wall'

    positive_score = reward_module.score(
        {"clean_image": image_tensor, "prompt": positive_prompt}
    )
    negative_score = reward_module.score(
        {"clean_image": image_tensor, "prompt": negative_prompt}
    )

    print(f"Positive (target='Hello World'): {positive_score.tolist()}")
    print(f"Negative (target='Goodbye Mars'): {negative_score.tolist()}")

    reward_module.unload_model()
