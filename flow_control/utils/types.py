from typing import Annotated, Any

import torch
from pydantic import BeforeValidator, PlainSerializer

from .ema import make_ema_optimizer


def validate_torch_dtype(v: Any) -> torch.dtype:
    if isinstance(v, torch.dtype):
        return v
    if isinstance(v, str):
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


TorchDType = Annotated[
    torch.dtype,
    BeforeValidator(validate_torch_dtype),
    PlainSerializer(serialize_torch_dtype, return_type=str),
]

TorchDevice = Annotated[
    torch.device,
    BeforeValidator(validate_torch_device),
    PlainSerializer(serialize_torch_device, return_type=str),
]


OptimizerConfig = dict[str, Any]


def parse_optimizer(conf: OptimizerConfig, parameters, ema_decay: float = 1.0):
    class_name = conf.pop("class_name")
    if class_name == "Prodigy":
        from prodigyopt import Prodigy

        ctor = Prodigy
    elif class_name.endswith("8bit") or class_name.startswith("Paged"):
        import bitsandbytes as bnb  # type: ignore

        ctor = getattr(bnb.optim, class_name)
    else:
        ctor = getattr(torch.optim, class_name)
    if ema_decay != 1.0:
        ctor = make_ema_optimizer(ctor)
        return ctor(parameters, ema_decay=ema_decay, **conf)
    else:
        return ctor(parameters, **conf)


SchedulerConfig = dict[str, Any]


def parse_scheduler(conf: SchedulerConfig, optimizer: torch.optim.Optimizer):
    class_name = conf.pop("class_name")
    ctor = getattr(torch.optim.lr_scheduler, class_name)
    return ctor(optimizer, **conf)
