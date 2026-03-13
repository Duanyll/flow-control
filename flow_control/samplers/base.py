from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.utils.common import deep_move_to_device
from flow_control.utils.logging import console, get_logger, warn_once

if TYPE_CHECKING:
    from .sampler import SampleOutput

logger = get_logger(__name__)


def make_sample_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description:<20}"),
        BarColumn(complete_style="blue", finished_style="bold blue"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


class BaseSampler(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cfg_scale: float = 7.5
    """
    TRUE classifier-free guidance scale. For guidance distilled models like FLUX, true
    CFG should not be applied and this should be set to 1.0. Set their guidance embeddings
    value in ModelAdapter instead.
    """
    seed: int = 42
    enable_cfg_renorm: bool = False
    cfg_renorm_eps: float = 1e-8
    cfg_renorm_min: float = 0.0

    _negative_pass: bool = False

    def _sync_negative_pass(self, negative_pass: bool):
        if dist.is_initialized():
            negative_pass_tensor = torch.tensor(negative_pass, device="cuda")
            dist.all_reduce(negative_pass_tensor, op=dist.ReduceOp.MAX)
            self._negative_pass = negative_pass_tensor.item() > 0
        else:
            self._negative_pass = negative_pass

    def sample(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start=1.0,
        t_end=0.0,
        return_trajectory: bool = False,
    ) -> SampleOutput:
        if self.cfg_scale > 1.0 and negative_batch is None:
            warn_once(
                logger,
                "cfg_scale > 1.0 but no negative_batch provided. This will disable classifier-free guidance.",
            )
        batch = deep_move_to_device(batch, model.device)
        if negative_batch is not None:
            negative_batch = deep_move_to_device(negative_batch, model.device)
        return self._sample(
            model,
            batch,
            negative_batch,
            t_start,
            t_end,
            return_trajectory=return_trajectory,
        )

    @abstractmethod
    def _sample(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start=1.0,
        t_end=0.0,
        return_trajectory: bool = False,
    ) -> SampleOutput:
        raise NotImplementedError()

    def get_guided_velocity(
        self,
        model: BaseModelAdapter,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Batch,
        negative_batch: Batch | None,
    ) -> torch.Tensor:
        # Sync per-call so CFG behavior is correct even when callers bypass sample().
        has_negative = self.cfg_scale > 1.0 and negative_batch is not None
        self._sync_negative_pass(has_negative)

        batch["noisy_latents"] = latents
        cond = model.predict_velocity(batch, timestep)
        if self._negative_pass:
            if negative_batch is not None:
                negative_batch["noisy_latents"] = latents
                uncond = model.predict_velocity(negative_batch, timestep)
                combined_velocity = uncond + (cond - uncond) * self.cfg_scale

                if self.enable_cfg_renorm:
                    cond_norm = torch.norm(cond, dim=2, keepdim=True)
                    noise_norm = torch.norm(combined_velocity, dim=2, keepdim=True)
                    combined_velocity = combined_velocity * (
                        cond_norm / (noise_norm + self.cfg_renorm_eps)
                    ).clamp(min=self.cfg_renorm_min, max=1.0)
                return combined_velocity
            else:
                # Must do an empty forward pass to sync with other processes
                _ = model.predict_velocity(batch, timestep)
                return cond
        else:
            return cond
