from typing import Annotated

from pydantic import PlainValidator

from .base import BaseModelAdapter, Batch
from .flux1 import parse_adapter as parse_flux1_adapter
from .longcat import parse_adapter as parse_longcat_adapter
from .qwen import parse_adapter as parse_qwen_adapter

MODEL_ADAPTER_PARSERS = {
    "flux1": parse_flux1_adapter,
    "qwen": parse_qwen_adapter,
    "longcat": parse_longcat_adapter,
}


def parse_model_adapter(conf: dict) -> BaseModelAdapter:
    model_type = conf["arch"]
    parser = MODEL_ADAPTER_PARSERS.get(model_type)
    if parser is None:
        raise ValueError(f"Unknown model adapter type: {model_type}")
    return parser(conf)


ModelAdapter = Annotated[BaseModelAdapter, PlainValidator(parse_model_adapter)]

__all__ = [
    "ModelAdapter",
    "Batch",
]
