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
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from flow_control.adapters import Batch, ModelAdapter
from flow_control.utils.logging import console, get_logger, warn_once

if TYPE_CHECKING:
    from .euler import SdeTrajectory

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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cfg_scale: float = 7.5
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
        model: ModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start=1.0,
        t_end=0.0,
    ) -> torch.Tensor:
        if self.cfg_scale > 1.0 and negative_batch is None:
            warn_once(
                logger,
                "cfg_scale > 1.0 but no negative_batch provided. This will disable classifier-free guidance.",
            )
        has_negative = self.cfg_scale > 1.0 and negative_batch is not None
        self._sync_negative_pass(has_negative)
        return self._sample(model, batch, negative_batch, t_start, t_end)

    @abstractmethod
    def _sample(
        self,
        model: ModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start=1.0,
        t_end=0.0,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def sample_with_logprob(
        self,
        model: ModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
    ) -> SdeTrajectory:
        """SDE sampling with log probability tracking. Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support sample_with_logprob. "
            "Use an SDE-capable sampler (e.g., ShiftedEulerSampler with noise_level > 0)."
        )

    def compute_logprob_at_step(
        self,
        model: ModelAdapter,
        batch: Batch,
        latent_t: torch.Tensor,
        latent_t_minus_1: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        negative_batch: Batch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log_prob of a known transition under current policy. Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support compute_logprob_at_step. "
            "Use an SDE-capable sampler (e.g., ShiftedEulerSampler with noise_level > 0)."
        )

    def get_guided_velocity(
        self,
        model: ModelAdapter,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Batch,
        negative_batch: Batch | None,
    ) -> torch.Tensor:
        dtype = batch["noisy_latents"].dtype
        batch["noisy_latents"] = latents.to(dtype)
        cond = model.predict_velocity(batch, timestep).float()
        if self._negative_pass:
            if negative_batch is not None:
                negative_batch["noisy_latents"] = latents.to(dtype)
                uncond = model.predict_velocity(negative_batch, timestep).float()
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
