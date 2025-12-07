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
from flow_control.utils.logging import console


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
