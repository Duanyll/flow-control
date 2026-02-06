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
    type: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg_scale: float = 7.5
    seed: int = 42
    enable_cfg_renorm: bool = False
    cfg_renorm_eps: float = 1e-8
    cfg_renorm_min: float = 0.0

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
        cond = model.predict_velocity(cast(Any, batch), timestep).float()
        if self.cfg_scale > 1.0:
            if negative_batch is not None:
                negative_batch["noisy_latents"] = latents.to(dtype)
                uncond = model.predict_velocity(
                    cast(Any, negative_batch), timestep
                ).float()
                combined_velocity = uncond + (cond - uncond) * self.cfg_scale
                if self.enable_cfg_renorm:
                    cond_norm = torch.norm(cond, dim=2, keepdim=True)
                    noise_norm = torch.norm(combined_velocity, dim=2, keepdim=True)
                    combined_velocity = combined_velocity * (
                        cond_norm / (noise_norm + self.cfg_renorm_eps)
                    ).clamp(min=self.cfg_renorm_min, max=1.0)
                return combined_velocity
            else:
                warn_once(
                    logger,
                    "CFG scale > 1.0 but no negative batch provided. Running without CFG.",
                )
                return cond
        else:
            return cond
