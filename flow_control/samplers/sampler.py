from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field
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
from flow_control.utils.logging import console, get_logger, warn_once
from flow_control.utils.tensor import deep_move_to_device

from .shift import NoShift, Shift
from .solver import FlowSolver, Solver, SolverState

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


@dataclass(slots=True)
class SampleOutput:
    final_latents: torch.Tensor
    latents: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    solver_states: list[SolverState | None] | None = None


class Sampler(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    steps: int = 50
    solver: Solver = Field(default_factory=FlowSolver)
    shift: Shift = Field(default_factory=NoShift)
    trajectory_window_size: int | None = None
    trajectory_window_range: tuple[int, int] | None = None

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

        sigmas = torch.linspace(t_start, t_end, self.steps + 1)
        sigmas = self.shift.apply(sigmas, batch, self.steps)
        return self._run_sampling_loop(
            model=model,
            batch=batch,
            sigmas=sigmas,
            negative_batch=negative_batch,
            return_trajectory=return_trajectory,
        )

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

    def _select_trajectory_window(self, num_timesteps: int) -> tuple[int, int]:
        if self.trajectory_window_size:
            range_start, range_end = self.trajectory_window_range or (
                0,
                num_timesteps - 1,
            )
            assert 0 <= range_start < range_end < num_timesteps, (
                "Invalid trajectory_window_range"
            )
            window_start = random.randint(
                range_start,
                range_end - self.trajectory_window_size,
            )
            return window_start, window_start + self.trajectory_window_size
        return 0, num_timesteps - 1

    def _run_sampling_loop(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        sigmas: torch.Tensor,
        negative_batch: Batch | None = None,
        return_trajectory: bool = False,
    ) -> SampleOutput:
        device = model.device
        dtype = model.dtype

        sigmas = sigmas.to(device=device, dtype=torch.float32)
        latents = batch["noisy_latents"].float()
        solver_state = self.solver.init_state(self.steps)

        train_start, train_end = self._select_trajectory_window(self.steps)
        selected_latents: list[torch.Tensor] | None = [] if return_trajectory else None
        all_log_probs: list[torch.Tensor] | None = [] if return_trajectory else None
        all_solver_states: list[SolverState | None] | None = (
            [] if return_trajectory else None
        )

        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=self.steps)
            for i in range(self.steps):
                sigma = sigmas[i : i + 1]
                sigma_next = sigmas[i + 1 : i + 2]
                velocity = self.get_guided_velocity(
                    model,
                    latents,
                    sigma,
                    batch,
                    negative_batch,
                )
                step_eta = self.solver.eta if train_start <= i < train_end else 0.0
                if return_trajectory and i == train_start:
                    assert selected_latents is not None
                    selected_latents.append(latents)
                if return_trajectory and train_start <= i < train_end:
                    assert all_solver_states is not None
                    all_solver_states.append(solver_state)

                step_result = self.solver.step(
                    velocity=velocity,
                    latents=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=step_eta,
                    state=solver_state,
                )
                latents = step_result.next_latents
                solver_state = step_result.state

                if return_trajectory and train_start <= i < train_end:
                    assert selected_latents is not None
                    assert all_log_probs is not None
                    selected_latents.append(latents)
                    all_log_probs.append(step_result.log_prob)

                progress.advance(task)

        output = SampleOutput(
            final_latents=latents.to(dtype),
            timesteps=sigmas[:-1].clone(),
        )
        if not return_trajectory:
            return output

        assert selected_latents is not None
        assert all_log_probs is not None
        assert all_solver_states is not None
        output.latents = torch.stack(selected_latents, dim=1)
        output.log_probs = torch.stack(all_log_probs, dim=1)
        output.timesteps = sigmas[train_start : train_end + 1]
        output.solver_states = all_solver_states
        return output

    def compute_logprob_at_step(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        latent_t: torch.Tensor,
        latent_next: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        negative_batch: Batch | None = None,
        solver_state: SolverState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.solver.supports_step_log_prob:
            msg = f"Solver '{self.solver.type}' does not support step-wise replay log-prob."
            raise ValueError(msg)
        velocity = self.get_guided_velocity(
            model,
            latent_t,
            sigma,
            batch,
            negative_batch,
        )
        step_result = self.solver.replay_step(
            velocity=velocity,
            latents=latent_t,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=latent_next,
            state=solver_state,
        )
        return step_result.log_prob, step_result.mean, step_result.std_dev
