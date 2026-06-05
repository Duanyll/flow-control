"""OCR-based reward.

Parses a target string from the prompt (text inside the first pair of double
quotes), runs OCR on the generated image, and scores the match between the
target and the OCR output as ``1 - levenshtein / len(target)`` clipped to
``[0, 1]``.

The OCR backend is built on top of `rapidocr` 3.x (PP-OCR ONNX wrapper, no
PaddlePaddle dependency).  When loaded on CUDA, the ONNX Runtime sessions are
configured to allocate through PyTorch's CUDA caching allocator so they can
coexist with the training model.
"""

from __future__ import annotations

import ctypes
import importlib
from contextlib import suppress
from typing import Any, Literal, Protocol

import torch
from PIL import Image
from pydantic import ConfigDict, PrivateAttr

from flow_control.utils import device as devutil
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import tensor_to_pil

from .base import BaseReward

logger = get_logger(__name__)


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device!s}.")

    if device.index is not None:
        return device.index

    return torch.cuda.current_device()


class _OrtTorchCudaAllocatorCallbacks:
    """C callback bridge from ONNX Runtime to PyTorch's CUDA allocator.

    ONNX Runtime accepts provider options containing raw function-pointer
    addresses for ``gpu_external_alloc`` / ``gpu_external_free`` /
    ``gpu_external_empty_cache``.  ``onnxruntime-training`` exposes first-party
    helpers for those addresses, but the lighter ``onnxruntime-gpu`` wheel used
    by this project does not.  These ``ctypes`` callbacks provide the same ABI
    and are kept alive by ``RapidOcrBackend`` for as long as the ORT sessions
    exist.
    """

    _AllocCallback = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    _FreeCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    _EmptyCacheCallback = ctypes.CFUNCTYPE(None)

    def __init__(self, device: torch.device) -> None:
        self._device = torch.device("cuda", _cuda_device_index(device))
        device_for_callback = self._device

        def alloc(size: int) -> int | None:
            try:
                with torch.cuda.device(device_for_callback):
                    ptr = torch.cuda.caching_allocator_alloc(
                        size, device=device_for_callback
                    )
                return int(ptr)
            except Exception:
                _OrtTorchCudaAllocatorCallbacks._log_callback_exception(
                    "PyTorch CUDA allocator failed while serving ONNX Runtime."
                )
                return None

        def free(ptr: int | None) -> None:
            if not ptr:
                return

            try:
                torch.cuda.caching_allocator_delete(ptr)
            except Exception:
                _OrtTorchCudaAllocatorCallbacks._log_callback_exception(
                    "PyTorch CUDA allocator failed to free ONNX Runtime allocation."
                )

        def empty_cache() -> None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                _OrtTorchCudaAllocatorCallbacks._log_callback_exception(
                    "PyTorch CUDA allocator failed to empty cache for ONNX Runtime."
                )

        self._alloc_cb = self._AllocCallback(alloc)
        self._free_cb = self._FreeCallback(free)
        self._empty_cache_cb = self._EmptyCacheCallback(empty_cache)

    @staticmethod
    def _callback_address(callback: Any) -> str:
        address = ctypes.cast(callback, ctypes.c_void_p).value
        if address is None:
            raise RuntimeError("Failed to obtain CUDA allocator callback address.")
        return str(address)

    def provider_options(self) -> dict[str, Any]:
        return {
            "device_id": self._device.index,
            "gpu_external_alloc": self._callback_address(self._alloc_cb),
            "gpu_external_free": self._callback_address(self._free_cb),
            "gpu_external_empty_cache": self._callback_address(self._empty_cache_cb),
        }

    @staticmethod
    def _log_callback_exception(message: str) -> None:
        with suppress(Exception):
            logger.exception(message)


def _ortmodule_torch_allocator_options(
    device: torch.device,
) -> tuple[dict[str, Any], Any] | None:
    """Return first-party ONNX Runtime allocator options when available."""
    try:
        torch_gpu_allocator = importlib.import_module(
            "onnxruntime.training.ortmodule.torch_cpp_extensions.torch_gpu_allocator"
        )
    except ModuleNotFoundError:
        return None

    device_id = _cuda_device_index(device)
    try:
        options = {
            "device_id": device_id,
            "gpu_external_alloc": str(
                torch_gpu_allocator.gpu_caching_allocator_raw_alloc_address()
            ),
            "gpu_external_free": str(
                torch_gpu_allocator.gpu_caching_allocator_raw_delete_address()
            ),
            "gpu_external_empty_cache": str(
                torch_gpu_allocator.gpu_caching_allocator_empty_cache_address()
            ),
        }
    except AttributeError:
        logger.warning(
            "onnxruntime.training allocator helpers are present but incomplete; "
            "falling back to ctypes CUDA allocator callbacks."
        )
        return None

    return options, torch_gpu_allocator


def _torch_cuda_allocator_options(device: torch.device) -> tuple[dict[str, Any], Any]:
    ortmodule_options = _ortmodule_torch_allocator_options(device)
    if ortmodule_options is not None:
        return ortmodule_options

    callbacks = _OrtTorchCudaAllocatorCallbacks(device)
    return callbacks.provider_options(), callbacks


def _rapidocr_params(
    use_cuda: bool, cuda_ep_cfg: dict[str, Any] | None
) -> dict[str, Any]:
    from rapidocr import EngineType, LangDet, LangRec

    params: dict[str, Any] = {
        # Force ONNX Runtime for every task (det / cls / rec).
        # rapidocr 3.x requires Enum instances for these keys.
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.EN,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.EN,
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
        # Pin ONNX Runtime to a small thread budget — by default it spawns
        # one intra-op thread per logical CPU and then tries to set
        # affinities, which fails noisily inside Slurm cpusets and
        # contends with the trainer's data loader workers.
        "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        # Keep rapidocr quiet during training.
        "Global.log_level": "error",
    }

    if cuda_ep_cfg is not None:
        for key, value in cuda_ep_cfg.items():
            params[f"EngineConfig.onnxruntime.cuda_ep_cfg.{key}"] = value

    return params


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
    When ``device`` is CUDA, RapidOCR's ONNX Runtime execution provider is
    asked to use PyTorch's CUDA caching allocator through the
    ``gpu_external_*`` provider options.  RapidOCR does not expose these keys
    as named constructor arguments, but its OmegaConf config accepts nested
    ``EngineConfig.onnxruntime.cuda_ep_cfg.*`` values, so we inject them there.

    If CUDA provider setup fails, we fall back to CPU OCR instead of letting
    ONNX Runtime reserve its own GPU arena and contend with the trainer.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        from rapidocr import RapidOCR

        self._cuda_allocator: Any | None = None

        cuda_device = device if device is not None and device.type == "cuda" else None
        use_cuda = cuda_device is not None
        cuda_ep_cfg: dict[str, Any] | None = None
        if cuda_device is not None and not torch.cuda.is_available():
            logger.warning(
                "CUDA OCR requested but no CUDA device is available; using CPU OCR."
            )
            use_cuda = False
        elif cuda_device is not None:
            try:
                cuda_ep_cfg, self._cuda_allocator = _torch_cuda_allocator_options(
                    cuda_device
                )
            except Exception:
                logger.exception(
                    "Could not wire ONNX Runtime to PyTorch CUDA allocator; "
                    "falling back to CPU OCR."
                )
                use_cuda = False

        params = _rapidocr_params(use_cuda=use_cuda, cuda_ep_cfg=cuda_ep_cfg)
        try:
            self._engine = RapidOCR(params=params)
        except Exception:
            if not use_cuda:
                raise

            logger.exception(
                "RapidOCR CUDA initialization failed even with PyTorch allocator; "
                "falling back to CPU OCR."
            )
            self._cuda_allocator = None
            self._engine = RapidOCR(
                params=_rapidocr_params(use_cuda=False, cuda_ep_cfg=None)
            )

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
    test_device = devutil.default_device()
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

    print(f"Positive (target='Hello World'): {positive_score.aggregate().tolist()}")
    print(f"Negative (target='Goodbye Mars'): {negative_score.aggregate().tolist()}")

    reward_module.unload_model()
