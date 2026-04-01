import torch
from pydantic import BaseModel, ConfigDict

from flow_control.adapters import ModelAdapter
from flow_control.processors import Processor
from flow_control.samplers import Sampler
from flow_control.utils.types import TorchDevice


class ServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 7860
    device: TorchDevice = torch.device("cuda")
    processor_device: TorchDevice | None = None
    offload_processor: bool = False
    share: bool = False

    model: ModelAdapter
    processor: Processor
    sampler: Sampler

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_root: str | None = None
    use_ema: bool = False
