"""Device-agnostic helpers for CUDA / Apple MPS / Ascend NPU.

The codebase historically hardcoded CUDA. These helpers route every device
operation through PyTorch's device-agnostic runtime so the same code runs on:

- ``cuda`` (NVIDIA, in-tree)
- ``mps`` (Apple Silicon, in-tree)
- ``npu`` (Huawei Ascend, out-of-tree ``torch_npu`` via the PrivateUse1 backend)

Design notes:

- **Resolution** (``default_device`` / ``current_device_type``) uses
  ``torch.accelerator`` — a pure, side-effect-free query that also routes the
  PrivateUse1 (NPU) backend, so it is safe to call from a Pydantic
  ``default_factory`` at config-instantiation time.
- **Per-device ops** (cache / seed / RNG) dispatch through
  ``torch.get_device_module(device)`` (→ ``torch.cuda`` / ``torch.mps`` /
  ``torch.npu``) rather than the ``torch.accelerator`` umbrella, because
  ``torch.accelerator.empty_cache()`` raises on MPS in torch 2.12
  (``Allocator for mps is not a DeviceAllocator``).
"""

from __future__ import annotations

from types import ModuleType

import torch
import torch.distributed as dist

__all__ = [
    "current_device_type",
    "default_device",
    "device_count",
    "dist_backend",
    "empty_cache",
    "get_rng_state",
    "is_available",
    "manual_seed_all",
    "set_device",
    "set_rng_state",
    "synchronize",
]


def default_device() -> torch.device:
    """Return the current accelerator, or CPU if none is available.

    ``cuda`` on an NVIDIA box, ``mps`` on Apple Silicon, ``npu`` with
    ``torch_npu`` installed, ``cpu`` otherwise.
    """
    acc = torch.accelerator.current_accelerator(check_available=True)
    return acc if acc is not None else torch.device("cpu")


def current_device_type() -> str:
    """Device-type string for the current accelerator (e.g. ``"cuda"``, ``"mps"``)."""
    acc = torch.accelerator.current_accelerator(check_available=True)
    return acc.type if acc is not None else "cpu"


def _device_module(device: torch.device | None = None) -> ModuleType:
    """The per-device runtime module (``torch.cuda`` / ``torch.mps`` / ...)."""
    if device is None:
        acc = torch.accelerator.current_accelerator(check_available=True)
        device = acc if acc is not None else torch.device("cpu")
    return torch.get_device_module(device)


def is_available() -> bool:
    """Whether an accelerator is built and usable at runtime."""
    return torch.accelerator.is_available()


def device_count() -> int:
    """Number of available accelerator devices (0 if none)."""
    return torch.accelerator.device_count() if is_available() else 0


def empty_cache(device: torch.device | None = None) -> None:
    """Release cached device memory; no-op when the device has no cache API."""
    fn = getattr(_device_module(device), "empty_cache", None)
    if fn is not None:
        fn()


def synchronize(device: torch.device | None = None) -> None:
    """Wait for all kernels on the device to complete; no-op on CPU."""
    fn = getattr(_device_module(device), "synchronize", None)
    if fn is not None:
        fn()


def set_device(device: torch.device | int) -> None:
    """Pin the active accelerator device index for this process; no-op on CPU."""
    if isinstance(device, torch.device):
        if device.type == "cpu":
            return
        index = device.index or 0
    else:
        index = int(device)
    torch.accelerator.set_device_index(index)


def manual_seed_all(seed: int) -> None:
    """Seed all accelerator RNGs (falls back to single-device ``manual_seed``)."""
    module = _device_module()
    fn = getattr(module, "manual_seed_all", None) or getattr(
        module, "manual_seed", None
    )
    if fn is not None:
        fn(seed)


def get_rng_state(device: torch.device) -> torch.Tensor | None:
    """RNG state for *device*, or ``None`` on CPU / when unsupported."""
    if device.type == "cpu":
        return None
    fn = getattr(_device_module(device), "get_rng_state", None)
    if fn is None:
        return None
    try:
        return fn(device)
    except TypeError:
        return fn()


def set_rng_state(state: torch.Tensor | None, device: torch.device) -> None:
    """Restore RNG state for *device*; no-op on CPU or when *state* is ``None``."""
    if device.type == "cpu" or state is None:
        return
    fn = getattr(_device_module(device), "set_rng_state", None)
    if fn is None:
        return
    try:
        fn(state, device)
    except TypeError:
        fn(state)


def dist_backend(device: torch.device) -> str:
    """Default ``torch.distributed`` backend for *device* (cuda→nccl, npu→hccl, else gloo)."""
    return dist.get_default_backend_for_device(device)


if __name__ == "__main__":
    from rich import print

    dev = default_device()
    print(f"default_device(): {dev}")
    print(f"current_device_type(): {current_device_type()}")
    print(f"is_available(): {is_available()}  device_count(): {device_count()}")
    print(f"dist_backend({dev}): {dist_backend(dev)}")

    # The key regression check: empty_cache must NOT raise on MPS.
    empty_cache()
    synchronize()
    manual_seed_all(0)
    state = get_rng_state(dev)
    set_rng_state(state, dev)
    print(f"rng round-trip ok (state is None: {state is None})")
    print("device helpers smoke test passed")
