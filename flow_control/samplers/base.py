from typing import Callable, cast, Any, TypeVar
import torch
from pydantic import BaseModel

from flow_control.adapters import BaseModelAdapter

T = TypeVar("T", bound=BaseModelAdapter)

class BaseSampler(BaseModel):
    cfg_scale: float = 7.5

    def sample(
        self,
        model: BaseModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1.0,
        t_end=0.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def get_guided_velocity(
        self,
        model: BaseModelAdapter,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: dict,
        negative_batch: dict | None,
    ) -> torch.Tensor:
        dtype = batch["noisy_latents"].dtype
        batch["noisy_latents"] = latents.to(dtype)
        velocity = model.predict_velocity(cast(Any, batch), timestep).float()
        if negative_batch is not None and self.cfg_scale > 1.0:
            negative_batch["noisy_latents"] = latents.to(dtype)
            negative_velocity = model.predict_velocity(
                cast(Any, negative_batch), timestep
            ).float()
            velocity = (
                negative_velocity + (velocity - negative_velocity) * self.cfg_scale
            )
        return velocity
