from abc import ABC, abstractmethod
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
from flow_control.utils.logging import console, get_logger, warn_once

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


class BaseSampler(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg_scale: float = 7.5
    seed: int = 42
    keep_cfg_norm: bool = False

    @abstractmethod
    def sample(
        self,
        model: ModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1.0,
        t_end=0.0,
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
        if self.cfg_scale > 1.0:
            if negative_batch is not None:
                negative_batch["noisy_latents"] = latents.to(dtype)
                negative_velocity = model.predict_velocity(
                    cast(Any, negative_batch), timestep
                ).float()
                combined_velocity = (
                    negative_velocity + (velocity - negative_velocity) * self.cfg_scale
                )
                if self.keep_cfg_norm:
                    velocity_norm = torch.norm(velocity, dim=2, keepdim=True)
                    combined_norm = torch.norm(combined_velocity, dim=2, keepdim=True)
                    combined_velocity = combined_velocity * (
                        velocity_norm / combined_norm
                    )
                return combined_velocity
            else:
                warn_once(
                    logger,
                    "CFG scale > 1.0 but no negative batch provided. Running without CFG.",
                )
                return velocity
        else:
            return velocity
