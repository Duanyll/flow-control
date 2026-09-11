"""GRPO's stochastic-step records and differentiable likelihood replay."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from flow_control.adapters.base import SamplerModel
from flow_control.samplers import Executor, Sampler, SampleRun, StepRecord
from flow_control.samplers.plan import Transition, euler_step
from flow_control.samplers.solver import (
    CPSSolver,
    DanceSolver,
    DDIMSolver,
    FlashSolver,
    FlowSolver,
)

_LIKELIHOOD_SOLVERS = (FlowSolver, DDIMSolver, CPSSolver, DanceSolver, FlashSolver)


@dataclass(slots=True)
class RecordedStep:
    latent_t: torch.Tensor
    latent_next: torch.Tensor
    log_prob: torch.Tensor
    transition: Transition
    item_index: int = 0
    num_items: int = 1
    noise_scale: float | None = None
    """Flash's executed ramp value, independent of later solver retuning."""


@dataclass(slots=True)
class ReplayItem:
    run: SampleRun
    """A training run over the rollout's executed plan (``Sampler.make_run(plan=...)``)."""
    recorded: RecordedStep


@dataclass(slots=True)
class StepLogProbOutput:
    log_prob: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor


def normal_log_prob(
    sample: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    sample, mean, scale = sample.float(), mean.float(), scale.float()
    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * scale**2)
        - torch.log(scale)
        - torch.log(
            torch.sqrt(
                torch.tensor(2 * math.pi, device=sample.device, dtype=sample.dtype)
            )
        )
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def step_log_prob(recorded: RecordedStep, velocity: torch.Tensor) -> StepLogProbOutput:
    """Rebuild the executed step's moments using its eta and ramp position.

    Flow retains its historical diffusion coefficient as ``std_dev`` for the
    GRPO KL objective; its Gaussian likelihood uses the coefficient times
    sqrt(-dt). CPS retains its unnormalized residual objective. Flash's
    likelihood is a Gaussian approximation when noise clipping is enabled.
    """
    tr = recorded.transition
    solver = tr.solver
    latents, velocity = recorded.latent_t.float(), velocity.float()
    if (tr.eta == 0.0 and not isinstance(solver, DDIMSolver)) or (
        isinstance(solver, FlashSolver) and tr.sigma_next <= 0.0
    ):
        return StepLogProbOutput(
            log_prob=torch.zeros(latents.shape[0], device=latents.device),
            mean=euler_step(latents, velocity, tr.sigma, tr.sigma_next),
            std_dev=latents.new_tensor(0.0),
        )
    if isinstance(solver, FlashSolver):
        mean, std_dev = solver.renoise_parts(
            latents,
            velocity,
            tr.sigma,
            tr.sigma_next,
            recorded.noise_scale
            if recorded.noise_scale is not None
            else solver.noise_scale_at(recorded.item_index, recorded.num_items),
        )
        noise_scale = std_dev
    elif isinstance(solver, FlowSolver):
        mean, std_dev, noise_scale = solver.step_parts(
            latents, velocity, tr.sigma, tr.sigma_next, tr.eta
        )
    elif isinstance(solver, (DDIMSolver, CPSSolver, DanceSolver)):
        mean, std_dev = solver.step_parts(
            latents, velocity, tr.sigma, tr.sigma_next, tr.eta
        )
        noise_scale = std_dev
    else:
        raise ValueError(f"GRPO has no step likelihood for solver {solver.type!r}.")
    if tr.eta == 0.0:
        log_prob = torch.zeros(latents.shape[0], device=latents.device)
    elif isinstance(solver, CPSSolver):
        residual = -((recorded.latent_next.detach().float() - mean) ** 2)
        log_prob = residual.mean(dim=tuple(range(1, residual.ndim)))
    else:
        log_prob = normal_log_prob(recorded.latent_next, mean, noise_scale)
    return StepLogProbOutput(log_prob, mean, std_dev)


@dataclass
class GrpoCollector:
    """Keep stochastic latents and likelihoods as steps finish; never retain velocities."""

    sampler: Sampler
    _records: dict[int, list[RecordedStep]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.sampler.guidance.init_state() is not None:
            raise ValueError(
                "GRPO requires stateless guidance; guidance-state replay is unsupported."
            )
        if not isinstance(self.sampler.solver, _LIKELIHOOD_SOLVERS):
            raise ValueError(
                f"GRPO has no step likelihood for solver {self.sampler.solver.type!r}."
            )

    def __call__(self, run: SampleRun, step: StepRecord) -> None:
        if step.transition.eta <= 0.0:
            return
        assert step.velocity is not None
        solver = step.transition.solver
        recorded = RecordedStep(
            step.latents,
            step.next_latents,
            step.next_latents.new_empty(0),
            step.transition,
            step.index,
            len(run.plan),
            noise_scale=(
                solver.noise_scale_at(step.index, len(run.plan))
                if isinstance(solver, FlashSolver)
                else None
            ),
        )
        recorded.log_prob = step_log_prob(recorded, step.velocity).log_prob
        self._records.setdefault(id(run), []).append(recorded)

    def take(self, run: SampleRun) -> list[RecordedStep]:
        records = self._records.pop(id(run), [])
        if not records:
            raise ValueError(
                "GRPO requires at least one stochastic step per sample; set solver.eta > 0 "
                "and keep a nonempty stochastic window in sampler.transforms."
            )
        return records


def replay_steps(
    model: SamplerModel, items: list[ReplayItem]
) -> list[StepLogProbOutput]:
    """Re-evaluate each recorded step's guided velocity and rebuild its likelihood."""
    if not items:
        raise ValueError("replay_steps requires at least one item.")
    executor = Executor(model, items[0].run.sampler.variant_keys())
    velocities = executor.evaluate(
        [
            item.run.guided_velocity(item.recorded.latent_t, item.recorded.item_index)
            for item in items
        ]
    )
    return [
        step_log_prob(item.recorded, velocity)
        for item, velocity in zip(items, velocities, strict=True)
    ]
