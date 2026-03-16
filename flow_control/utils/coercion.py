"""Pydantic-based coercion utilities for dataset loading.

Provides Annotated type aliases with BeforeValidator hooks that automatically
convert raw string data (e.g. from CSV) into the expected Python/Torch types.
"""

import json
import os
from functools import lru_cache
from typing import Annotated, Any, Literal

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from pydantic import BeforeValidator, TypeAdapter, ValidationInfo

from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import pil_to_tensor

logger = get_logger(__name__)


# ----------------------------- Image Loading ------------------------------ #


def _load_attachment(path: str) -> torch.Tensor:
    """Load a file as a torch.Tensor. Supports .npy/.npz, .pt/.pth, and images."""
    extname = os.path.splitext(path)[1].lower()
    if extname in (".npy", ".npz"):
        return torch.from_numpy(np.load(path)).clone()
    elif extname in (".pt", ".pth"):
        return torch.load(path, weights_only=True)
    else:
        image = Image.open(path)
        return pil_to_tensor(image)


def _resolve_path(value: str, info: ValidationInfo) -> str:
    """Resolve a relative path using attachment_dir from validation context."""
    if os.path.isabs(value):
        return value
    ctx = info.context
    if ctx and "attachment_dir" in ctx:
        return os.path.join(ctx["attachment_dir"], value)
    return value


# ------------------------------ Validators -------------------------------- #


def _coerce_to_image_tensor(value: Any) -> torch.Tensor:
    """Coerce a single image-like value to a 1CHW float tensor in [0, 1].

    Accepts:
    - torch.Tensor: 1CHW, CHW, 1HWC, or HWC; float (assumed [0,1]) or uint8 (scaled)
    - PIL.Image.Image: any mode
    - np.ndarray: HWC or CHW uint8 or float
    """
    if isinstance(value, Image.Image):
        return pil_to_tensor(value)

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).clone()

    if isinstance(value, torch.Tensor):
        t = value
        if t.ndim == 4:
            # Could be 1CHW or 1HWC
            if t.shape[-1] in (1, 2, 3, 4) and t.shape[1] not in (1, 2, 3, 4):
                t = rearrange(t, "b h w c -> b c h w")
        elif t.ndim == 3:
            if t.shape[-1] in (1, 2, 3, 4) and t.shape[0] not in (1, 2, 3, 4):
                t = rearrange(t, "h w c -> 1 c h w")
            else:
                t = t.unsqueeze(0)  # CHW -> 1CHW
        else:
            raise ValueError(f"Unsupported tensor shape {value.shape} for image tensor")
        t = t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
        return t

    raise ValueError(f"Cannot coerce {type(value)} to image tensor")


def _validate_image_tensor(value: Any, info: ValidationInfo) -> torch.Tensor:
    if isinstance(value, str):
        path = _resolve_path(value, info)
        return _load_attachment(path)
    return _coerce_to_image_tensor(value)


def _validate_image_tensor_list(value: Any, info: ValidationInfo) -> list[torch.Tensor]:
    if isinstance(value, str):
        paths = [p.strip() for p in value.split(";") if p.strip()]
        return [_load_attachment(_resolve_path(p, info)) for p in paths]
    if isinstance(value, list):
        result = []
        for v in value:
            if isinstance(v, str):
                result.append(_load_attachment(_resolve_path(v, info)))
            else:
                result.append(_coerce_to_image_tensor(v))
        return result
    raise ValueError(
        f"Expected list[Tensor|PIL|ndarray], list[str], or str, got {type(value)}"
    )


def _validate_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {value}") from e
    return value


def _validate_json_str_list(value: Any) -> list[str]:
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return [v.strip() for v in value.split(";") if v.strip()]
    raise ValueError(f"Expected list[str] or str, got {type(value)}")


# ----------------------------- Type Aliases ------------------------------- #

ImageTensor = Annotated[torch.Tensor, BeforeValidator(_validate_image_tensor)]
ImageTensorList = Annotated[
    list[torch.Tensor], BeforeValidator(_validate_image_tensor_list)
]
JsonStrList = Annotated[list[str], BeforeValidator(_validate_json_str_list)]
JsonBeforeValidator = BeforeValidator(_validate_json)

# ----------------------------- Utility Functions -------------------------- #


def get_input_typeddict(
    processor_class: type, mode: Literal["training", "inference"]
) -> type | None:
    """Extract the TypedDict type from a processor class's generic parameters.

    Returns the InputBatch (mode="inference") or TrainInputBatch (mode="training")
    TypedDict class, or None if it cannot be determined.
    """
    from flow_control.processors.base import BaseProcessor

    # Pydantic stores resolved generic args in __pydantic_generic_metadata__
    # on the parameterized BaseProcessor[...] class in the MRO.
    for base in processor_class.__mro__:
        meta = getattr(base, "__pydantic_generic_metadata__", None)
        if meta is None:
            continue
        origin = meta.get("origin")
        args = meta.get("args", ())
        if (
            origin is not None
            and (origin is BaseProcessor or issubclass(origin, BaseProcessor))
            and len(args) >= 2
        ):
            if mode == "inference":
                return args[0]
            else:
                return args[1]
    return None


@lru_cache
def build_type_adapter(typed_dict_class: type) -> TypeAdapter:
    """Create and cache a TypeAdapter for the given TypedDict class."""
    return TypeAdapter(typed_dict_class)


def coerce_record(
    raw: dict[str, Any],
    adapter: TypeAdapter,
    attachment_dir: str,
) -> dict[str, Any]:
    """Validate and convert a raw record using the TypeAdapter.

    Fields not present in the TypedDict are passed through unchanged.
    """
    return adapter.validate_python(
        raw, context={"attachment_dir": attachment_dir}, extra="allow"
    )


if __name__ == "__main__":
    # Test basic coercion types
    from pydantic import TypeAdapter as TA
    from rich import print

    # Test JsonStrList
    adapter = TA(JsonStrList)
    assert adapter.validate_python(["a", "b"]) == ["a", "b"]
    assert adapter.validate_python('["a","b"]') == ["a", "b"]
    assert adapter.validate_python("a;b") == ["a", "b"]
    print("[green]JsonStrList: OK[/green]")

    # Test get_input_typeddict
    from flow_control.processors.tasks.t2i import T2IProcessor

    td = get_input_typeddict(T2IProcessor, "training")
    print(f"T2IProcessor training TypedDict: {td}")
    assert td is not None

    td = get_input_typeddict(T2IProcessor, "inference")
    print(f"T2IProcessor inference TypedDict: {td}")
    assert td is not None

    # Test build_type_adapter with a simple TypedDict
    from typing import TypedDict

    class TestBatch(TypedDict):
        x: Annotated[int, JsonBeforeValidator]
        y: Annotated[float, JsonBeforeValidator]

    ta = build_type_adapter(TestBatch)
    result = ta.validate_python({"x": "10", "y": "3.14"})
    assert result == {"x": 10, "y": 3.14}
    print(f"[green]TypeAdapter coercion: {result}[/green]")

    print("[bold green]All tests passed![/bold green]")
