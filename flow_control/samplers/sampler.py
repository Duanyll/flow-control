from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

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
from flow_control.utils.progress import report_progress
from flow_control.utils.tensor import deep_move_to_device

from .shift import NoShift, Shift
from .solver import FlowSolver, SASolver, Solver, SolverState

logger = get_logger(__name__)


def derive_seed(base_seed: int, key: str) -> int:
    """Derive a deterministic per-sample seed from a base seed and a sample key."""
    h = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63)


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
class SampleRequest:
    batch: Batch
    negative_batch: Batch | None = None
    generator: torch.Generator | None = None


@dataclass(slots=True)
class SampleOutput:
    final_latents: torch.Tensor
    latents: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    solver_states: list[SolverState | None] | None = None


@dataclass(slots=True)
class ReplayRequest:
    batch: Batch
    latent_t: torch.Tensor
    latent_next: torch.Tensor
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    negative_batch: Batch | None = None
    solver_state: SolverState | None = None


@dataclass(slots=True)
class StepLogProbOutput:
    log_prob: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor


@dataclass(slots=True)
class _SamplingState:
    request: SampleRequest
    sigmas: torch.Tensor
    latents: torch.Tensor
    solver_state: SolverState | None
    train_start: int
    train_end: int
    selected_latents: list[torch.Tensor] | None
    log_probs: list[torch.Tensor] | None
    solver_states: list[SolverState | None] | None


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
    sigma_schedule: Literal["linear", "diffusers_flow"] = "linear"
    """Sigma-grid construction. ``diffusers_flow`` reproduces
    ``FlowMatchEulerDiscreteScheduler.set_timesteps`` including its shifted
    training-grid endpoints."""
    num_train_timesteps: int = 1000
    custom_sigmas: list[float] | None = None
    """Explicit sigma grid of length ``steps + 1`` (descending, terminal usually
    0.0), e.g. a distilled model's official timestep table. When set it replaces
    the linspace grid and *bypasses* ``shift`` (and the ``t_start``/``t_end``
    arguments of :meth:`sample`)."""
    trajectory_window_size: int | None = None
    trajectory_window_range: tuple[int, int] | None = None

    _negative_pass: bool = False

    @staticmethod
    def _validate_distributed_request_count(
        count: int,
        device: torch.device,
        operation: str,
    ) -> None:
        if not dist.is_initialized():
            return
        counts = torch.tensor([count, -count], device=device, dtype=torch.int64)
        dist.all_reduce(counts, op=dist.ReduceOp.MIN)
        if int(counts[0].item()) != -int(counts[1].item()):
            raise ValueError(
                f"All distributed ranks must submit the same number of requests "
                f"to Sampler.{operation}."
            )

    def _sync_negative_pass(
        self,
        negative_pass: bool,
        device: torch.device,
    ) -> None:
        if dist.is_initialized():
            negative_pass_tensor = torch.tensor(
                int(negative_pass), device=device, dtype=torch.int64
            )
            dist.all_reduce(negative_pass_tensor, op=dist.ReduceOp.MAX)
            self._negative_pass = bool(negative_pass_tensor.item())
        else:
            self._negative_pass = negative_pass

    def _make_sigmas(
        self,
        batch: Batch,
        t_start: float,
        t_end: float,
    ) -> torch.Tensor:
        if self.custom_sigmas is not None:
            # Serving may mutate steps at runtime, so validate this per call.
            if len(self.custom_sigmas) != self.steps + 1:
                raise ValueError(
                    f"custom_sigmas must have steps + 1 = {self.steps + 1} entries, "
                    f"got {len(self.custom_sigmas)}."
                )
            return torch.tensor(self.custom_sigmas, dtype=torch.float32)

        if self.sigma_schedule == "diffusers_flow":
            if t_start != 1.0 or t_end != 0.0:
                raise ValueError(
                    "diffusers_flow sigma schedule currently requires "
                    "t_start=1.0 and t_end=0.0."
                )
            training_grid = torch.linspace(
                1.0,
                1.0 / self.num_train_timesteps,
                self.num_train_timesteps,
            )
            shifted_training_grid = self.shift.apply(
                training_grid, batch, self.num_train_timesteps
            )
            inference_grid = torch.linspace(
                shifted_training_grid[0], shifted_training_grid[-1], self.steps
            )
            inference_grid = self.shift.apply(inference_grid, batch, self.steps)
            return torch.cat([inference_grid, inference_grid.new_zeros(1)])

        sigmas = torch.linspace(t_start, t_end, self.steps + 1)
        return self.shift.apply(sigmas, batch, self.steps)

    def sample(
        self,
        model: BaseModelAdapter,
        requests: list[SampleRequest],
        t_start: float = 1.0,
        t_end: float = 0.0,
        return_trajectory: bool = False,
    ) -> list[SampleOutput]:
        if not requests:
            raise ValueError("sample requires at least one request.")
        self._validate_distributed_request_count(len(requests), model.device, "sample")

        if self.cfg_scale > 1.0 and any(
            request.negative_batch is None for request in requests
        ):
            warn_once(
                logger,
                "cfg_scale > 1.0 but at least one request has no negative_batch; "
                "classifier-free guidance is disabled for those requests.",
            )

        prepared_requests = [
            SampleRequest(
                batch=deep_move_to_device(request.batch, model.device),
                negative_batch=(
                    deep_move_to_device(request.negative_batch, model.device)
                    if request.negative_batch is not None
                    else None
                ),
                generator=request.generator,
            )
            for request in requests
        ]
        sigmas = [
            self._make_sigmas(request.batch, t_start, t_end).to(
                device=model.device, dtype=torch.float32
            )
            for request in prepared_requests
        ]

        if isinstance(self.solver, SASolver):
            if len(requests) > 1:
                warn_once(
                    logger,
                    "SA-Solver does not expose a step-wise state API; requests "
                    "will be sampled sequentially.",
                )
            return [
                self._run_sa_sampling_loop(
                    model=model,
                    request=request,
                    sigmas=request_sigmas,
                    return_trajectory=return_trajectory,
                )
                for request, request_sigmas in zip(
                    prepared_requests, sigmas, strict=True
                )
            ]

        states = [
            self._init_sampling_state(request, request_sigmas, return_trajectory)
            for request, request_sigmas in zip(prepared_requests, sigmas, strict=True)
        ]
        self._run_sampling_loop(model, states, return_trajectory)
        return [
            self._make_sample_output(state, model.dtype, return_trajectory)
            for state in states
        ]

    def get_guided_velocity(
        self,
        model: BaseModelAdapter,
        batches: list[Batch],
        negative_batches: list[Batch | None],
        latents: list[torch.Tensor],
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        lengths = {
            len(batches),
            len(negative_batches),
            len(latents),
            len(timesteps),
        }
        if lengths == {0}:
            raise ValueError("get_guided_velocity requires at least one sample.")
        if len(lengths) != 1:
            raise ValueError(
                "batches, negative_batches, latents, and timesteps must have "
                "equal lengths."
            )

        conditional_batches: list[Batch] = []
        for batch, latent in zip(batches, latents, strict=True):
            conditional_batch = batch.copy()
            conditional_batch["noisy_latents"] = latent
            conditional_batches.append(conditional_batch)

        has_negative = [
            self.cfg_scale > 1.0 and negative_batch is not None
            for negative_batch in negative_batches
        ]
        self._sync_negative_pass(any(has_negative), model.device)

        conditional = model.predict_velocity_batched(conditional_batches, timesteps)
        if not self._negative_pass:
            return conditional

        unconditional_batches: list[Batch] = []
        for conditional_batch, negative_batch, latent, use_negative in zip(
            conditional_batches,
            negative_batches,
            latents,
            has_negative,
            strict=True,
        ):
            if use_negative:
                assert negative_batch is not None
                unconditional_batch = negative_batch.copy()
                unconditional_batch["noisy_latents"] = latent
                unconditional_batches.append(unconditional_batch)
            else:
                # A real forward keeps FSDP collectives aligned across ranks. Its
                # output is intentionally ignored for samples without CFG.
                unconditional_batches.append(conditional_batch)

        unconditional = model.predict_velocity_batched(unconditional_batches, timesteps)
        guided: list[torch.Tensor] = []
        for cond, uncond, use_negative in zip(
            conditional, unconditional, has_negative, strict=True
        ):
            if not use_negative:
                guided.append(cond)
                continue

            combined = uncond + (cond - uncond) * self.cfg_scale
            if self.enable_cfg_renorm:
                cond_norm = torch.norm(cond, dim=2, keepdim=True)
                noise_norm = torch.norm(combined, dim=2, keepdim=True)
                combined = combined * (
                    cond_norm / (noise_norm + self.cfg_renorm_eps)
                ).clamp(min=self.cfg_renorm_min, max=1.0)
            guided.append(combined)
        return guided

    def _select_trajectory_window(
        self,
        num_timesteps: int,
        generator: torch.Generator | None,
    ) -> tuple[int, int]:
        if self.trajectory_window_size is None:
            return 0, num_timesteps - 1

        window_size = self.trajectory_window_size
        if window_size <= 0:
            raise ValueError("trajectory_window_size must be positive.")
        range_start, range_end = self.trajectory_window_range or (
            0,
            num_timesteps - 1,
        )
        if not 0 <= range_start < range_end < num_timesteps:
            raise ValueError(
                "trajectory_window_range must satisfy 0 <= start < end < num_timesteps."
            )
        max_start = range_end - window_size
        if max_start < range_start:
            raise ValueError(
                f"trajectory_window_size={window_size} does not fit in "
                f"trajectory_window_range=({range_start}, {range_end})."
            )

        random_device = generator.device if generator is not None else "cpu"
        window_start = int(
            torch.randint(
                range_start,
                max_start + 1,
                (),
                generator=generator,
                device=random_device,
            ).item()
        )
        return window_start, window_start + window_size

    def _init_sampling_state(
        self,
        request: SampleRequest,
        sigmas: torch.Tensor,
        return_trajectory: bool,
    ) -> _SamplingState:
        train_start, train_end = self._select_trajectory_window(
            self.steps, request.generator
        )
        return _SamplingState(
            request=request,
            sigmas=sigmas,
            latents=request.batch["noisy_latents"].float(),
            solver_state=self.solver.init_state(self.steps),
            train_start=train_start,
            train_end=train_end,
            selected_latents=[] if return_trajectory else None,
            log_probs=[] if return_trajectory else None,
            solver_states=[] if return_trajectory else None,
        )

    def _run_sampling_loop(
        self,
        model: BaseModelAdapter,
        states: list[_SamplingState],
        return_trajectory: bool,
    ) -> None:
        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=self.steps)
            for step in range(self.steps):
                velocities = self.get_guided_velocity(
                    model=model,
                    batches=[state.request.batch for state in states],
                    negative_batches=[state.request.negative_batch for state in states],
                    latents=[state.latents for state in states],
                    timesteps=[state.sigmas[step : step + 1] for state in states],
                )
                for state, velocity in zip(states, velocities, strict=True):
                    sigma = state.sigmas[step : step + 1]
                    sigma_next = state.sigmas[step + 1 : step + 2]
                    step_eta = (
                        self.solver.eta
                        if state.train_start <= step < state.train_end
                        else 0.0
                    )
                    if return_trajectory and step == state.train_start:
                        assert state.selected_latents is not None
                        state.selected_latents.append(state.latents)
                    if (
                        return_trajectory
                        and state.train_start <= step < state.train_end
                    ):
                        assert state.solver_states is not None
                        state.solver_states.append(state.solver_state)

                    step_result = self.solver.step(
                        velocity=velocity,
                        latents=state.latents,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        eta=step_eta,
                        state=state.solver_state,
                        generator=state.request.generator,
                    )
                    state.latents = step_result.next_latents
                    state.solver_state = step_result.state

                    if (
                        return_trajectory
                        and state.train_start <= step < state.train_end
                    ):
                        assert state.selected_latents is not None
                        assert state.log_probs is not None
                        state.selected_latents.append(state.latents)
                        state.log_probs.append(step_result.log_prob)

                progress.advance(task)
                report_progress(
                    (step + 1) / self.steps,
                    f"Sampling {step + 1}/{self.steps}",
                )

    @staticmethod
    def _make_sample_output(
        state: _SamplingState,
        dtype: torch.dtype,
        return_trajectory: bool,
    ) -> SampleOutput:
        output = SampleOutput(
            final_latents=state.latents.to(dtype),
            timesteps=state.sigmas[:-1].clone(),
        )
        if not return_trajectory:
            return output

        assert state.selected_latents is not None
        assert state.log_probs is not None
        assert state.solver_states is not None
        output.latents = torch.stack(state.selected_latents, dim=1)
        output.log_probs = torch.stack(state.log_probs, dim=1)
        output.timesteps = state.sigmas[state.train_start : state.train_end + 1]
        output.solver_states = state.solver_states
        return output

    def _run_sa_sampling_loop(
        self,
        model: BaseModelAdapter,
        request: SampleRequest,
        sigmas: torch.Tensor,
        return_trajectory: bool,
    ) -> SampleOutput:
        assert isinstance(self.solver, SASolver)
        latents = request.batch["noisy_latents"].float()

        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=self.steps)

            def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                timestep = t.reshape(1).expand(x.shape[0])
                return self.get_guided_velocity(
                    model=model,
                    batches=[request.batch],
                    negative_batches=[request.negative_batch],
                    latents=[x],
                    timesteps=[timestep],
                )[0]

            def report_step(step: int) -> None:
                progress.advance(task)
                report_progress(
                    step / self.steps,
                    f"Sampling {step}/{self.steps}",
                )

            final_latents, intermediate_latents = self.solver.sample(
                model_fn=model_fn,
                latents=latents,
                sigmas=sigmas,
                generator=request.generator,
                step_callback=report_step,
            )

        return SampleOutput(
            final_latents=final_latents.to(model.dtype),
            latents=(
                torch.stack(intermediate_latents, dim=1) if return_trajectory else None
            ),
            timesteps=sigmas[:-1].clone(),
        )

    def compute_logprob_at_step(
        self,
        model: BaseModelAdapter,
        requests: list[ReplayRequest],
    ) -> list[StepLogProbOutput]:
        if not requests:
            raise ValueError("compute_logprob_at_step requires at least one request.")
        self._validate_distributed_request_count(
            len(requests), model.device, "compute_logprob_at_step"
        )
        if not self.solver.supports_step_log_prob:
            msg = f"Solver '{self.solver.type}' does not support step-wise replay log-prob."
            raise ValueError(msg)

        velocities = self.get_guided_velocity(
            model=model,
            batches=[request.batch for request in requests],
            negative_batches=[request.negative_batch for request in requests],
            latents=[request.latent_t for request in requests],
            timesteps=[request.sigma for request in requests],
        )
        outputs: list[StepLogProbOutput] = []
        for request, velocity in zip(requests, velocities, strict=True):
            step_result = self.solver.replay_step(
                velocity=velocity,
                latents=request.latent_t,
                sigma=request.sigma,
                sigma_next=request.sigma_next,
                prev_sample=request.latent_next,
                state=request.solver_state,
            )
            outputs.append(
                StepLogProbOutput(
                    log_prob=step_result.log_prob,
                    mean=step_result.mean,
                    std_dev=step_result.std_dev,
                )
            )
        return outputs
