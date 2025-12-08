from typing import Any, cast

import torch
from pydantic import BaseModel, ConfigDict
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from flow_control.adapters import ModelAdapter
from flow_control.utils.logging import console, get_logger

logger = get_logger(__name__)

def make_sample_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="blue", finished_style="bold blue"),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


class BaseSampler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg_scale: float = 7.5
    seed: int = 42

    def sample(
        self,
        model: ModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1.0,
        t_end=0.0
    ) -> torch.Tensor:
        raise NotImplementedError()

    def get_guided_velocity(
        self,
        model: ModelAdapter,
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

    def _init_noise_maybe(self, model: ModelAdapter, batch: dict, t_start: float):
        device = model.device
        dtype = model.dtype
        if "noisy_latents" not in batch:
            generator = torch.Generator(device=device).manual_seed(self.seed)
            timestep = torch.tensor([t_start], device=device, dtype=dtype)
            noise = model.generate_noise(
                batch, generator=generator # type: ignore
            )
            if "clean_latents" in batch:
                clean = batch["clean_latents"]
                batch["noisy_latents"] = (1.0 - timestep) * clean + timestep * noise
            else:
                batch["noisy_latents"] = noise
                if t_start < 1.0:
                    logger.warning(
                        "t_start < 1.0 but no clean_latents provided. "
                        "Sampling may not work as expected."
                    )