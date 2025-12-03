import torch
from typing import Any, Annotated
from pydantic import BeforeValidator, PlainSerializer


def validate_torch_dtype(v: Any) -> torch.dtype:
    if isinstance(v, torch.dtype):
        return v
    if isinstance(v, str):
        # 尝试从 torch 模块获取属性，例如 "float32" -> torch.float32
        # 注意：这里做了一个检查，确保获取到的是 dtype 而不是 torch 的其他函数
        try:
            dtype_obj = getattr(torch, v)
            if isinstance(dtype_obj, torch.dtype):
                return dtype_obj
        except AttributeError:
            pass
    raise ValueError(f"Invalid torch dtype: {v}")

def serialize_torch_dtype(v: torch.dtype) -> str:
    return str(v).split(".")[-1]  # 将 torch.float32 转为 "float32"

def validate_torch_device(v: Any) -> torch.device:
    if isinstance(v, torch.device):
        return v
    if isinstance(v, (str, int)):
        return torch.device(v)
    raise ValueError(f"Invalid torch device: {v}")

def serialize_torch_device(v: torch.device) -> str:
    return str(v)

# --- 2. 定义 Pydantic 自定义类型 ---

# TorchDType: 接受 str 或 torch.dtype，输出 torch.dtype，序列化为 str
TorchDType = Annotated[
    torch.dtype,
    BeforeValidator(validate_torch_dtype),
    PlainSerializer(serialize_torch_dtype, return_type=str),
]

# TorchDevice: 接受 str, int 或 torch.device，输出 torch.device，序列化为 str
TorchDevice = Annotated[
    torch.device,
    BeforeValidator(validate_torch_device),
    PlainSerializer(serialize_torch_device, return_type=str),
]