from typing import Annotated

from pydantic import PlainValidator

from .base import BaseModelAdapter
from .flux1 import parse_adapter as parse_flux1_adapter

MODEL_ADAPTER_PARSERS = {
    "flux1": parse_flux1_adapter,
}

def parse_model_adapter(conf: dict):
    model_type = conf.pop("model")
    parser = MODEL_ADAPTER_PARSERS.get(model_type)
    if parser is None:
        raise ValueError(f"Unknown model adapter type: {model_type}")
    return parser(conf)

ModelAdapter = Annotated[BaseModelAdapter, PlainValidator(parse_model_adapter)]

__all__ = [
    "ModelAdapter",
]