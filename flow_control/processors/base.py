from typing import Literal, TypedDict
import torch
from pydantic import BaseModel

from flow_control.utils.types import TorchDevice


class BaseProcessor(BaseModel):
    class BatchType(TypedDict):
        pass

    _loading_preset: dict[str, list[Literal["encode", "decode", "always"]]] = {}
    device: TorchDevice = torch.device("cuda")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for field_name, preset in self._loading_preset.items():
            if "always" in preset:
                model = getattr(self, field_name).load_model()
                setattr(self, f"_{field_name}", model)
            
    def load_models(self, preset: list[Literal["encode", "decode"]], device: torch.device | None = None) -> None:
        if device is None:
            device = self.device
        for field_name, field_preset in self._loading_preset.items():
            if any(p in field_preset for p in preset):
                model = getattr(self, field_name).load_model()
                model.to(device)
                setattr(self, f"_{field_name}", model)
            elif "always" not in field_preset:
                setattr(self, f"_{field_name}", None)

    def preprocess_batch(self, batch: BatchType) -> BatchType:
        raise NotImplementedError()
    
    def make_negative_batch(self, batch: BatchType) -> BatchType:
        raise NotImplementedError()
    
    def decode_output(self, output_latent: torch.Tensor, batch: BatchType) -> torch.Tensor:
        raise NotImplementedError()