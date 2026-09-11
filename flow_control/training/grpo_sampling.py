"""GRPO's stochastic-step collection and differentiable likelihood replay."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from flow_control.adapters.base import Batch, SamplerModel
from flow_control.samplers.evaluation import evaluate
from flow_control.samplers.executor import (
    StepObserver,
    validate_distributed_request_count,
)
from flow_control.samplers.plan import EvalRequest, StepContext, Transition, euler_step
from flow_control.samplers.sampler import SampleOutput, Sampler, SampleRequest
from flow_control.samplers.solver import (
    CPSSolver,
    DanceSolver,
    DDIMSolver,
    FlashSolver,
    FlowSolver,
)


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
    batch: Batch
    recorded: RecordedStep
    negative_batch: Batch | None = None


@dataclass(slots=True)
class StepLogProbOutput:
    log_prob: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor


def normal_log_prob(
    sample: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
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
    latents = recorded.latent_t
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
        residual = -((recorded.latent_next.detach() - mean) ** 2)
        log_prob = residual.mean(dim=tuple(range(1, residual.ndim)))
    else:
        log_prob = normal_log_prob(recorded.latent_next, mean, noise_scale)
    return StepLogProbOutput(log_prob, mean, std_dev)


def _validate_guidance(sampler: Sampler) -> None:
    if sampler.guidance.init_state() is not None:
        raise ValueError(
            "GRPO requires stateless guidance; guidance-state replay is unsupported."
        )


def collect_samples(
    sampler: Sampler,
    model: SamplerModel,
    requests: list[SampleRequest],
    *,
    observer: StepObserver | None = None,
) -> tuple[list[SampleOutput], list[list[RecordedStep]]]:
    _validate_guidance(sampler)
    if not isinstance(
        sampler.solver, (FlowSolver, DDIMSolver, CPSSolver, DanceSolver, FlashSolver)
    ):
        raise ValueError(
            f"GRPO has no step likelihood for solver {sampler.solver.type!r}."
        )
    trajectories: list[list[RecordedStep]] = [[] for _ in requests]

    def collect(
        run_index: int,
        transition: Transition,
        ctx: StepContext,
        velocity: torch.Tensor | None,
        next_latents: torch.Tensor,
    ) -> None:
        if transition.eta > 0.0:
            assert velocity is not None
            recorded = RecordedStep(
                ctx.latents,
                next_latents,
                next_latents.new_empty(0),
                transition,
                ctx.item_index,
                ctx.num_items,
                noise_scale=(
                    transition.solver.noise_scale_at(ctx.item_index, ctx.num_items)
                    if isinstance(transition.solver, FlashSolver)
                    else None
                ),
            )
            recorded.log_prob = step_log_prob(recorded, velocity).log_prob
            trajectories[run_index].append(recorded)
        if observer is not None:
            observer(run_index, transition, ctx, velocity, next_latents)

    outputs = sampler.sample(model, requests, observer=collect)
    if any(not steps for steps in trajectories):
        raise ValueError(
            "GRPO requires at least one stochastic step per sample; set solver.eta > 0 "
            "and keep a nonempty stochastic window in sampler.transforms."
        )
    return outputs, trajectories


def replay_steps(
    sampler: Sampler, model: SamplerModel, items: list[ReplayItem]
) -> list[StepLogProbOutput]:
    if not items:
        raise ValueError("replay_steps requires at least one item.")
    _validate_guidance(sampler)
    validate_distributed_request_count(len(items), model.device, "GRPO.replay_steps")
    requests = [
        EvalRequest(
            latents=item.recorded.latent_t,
            sigma=item.recorded.transition.sigma,
            sigma_next=item.recorded.transition.sigma_next,
            eta=item.recorded.transition.eta,
            solver=item.recorded.transition.solver,
        )
        for item in items
    ]
    contexts = [
        StepContext(
            latents=item.recorded.latent_t,
            generator=None,
            solver_state=None,
            guidance_state=None,
            item_index=item.recorded.item_index,
            num_items=item.recorded.num_items,
        )
        for item in items
    ]
    outputs = evaluate(
        model=model,
        guidance=sampler.guidance,
        batches=[item.batch for item in items],
        negative_batches=[item.negative_batch for item in items],
        requests=requests,
        contexts=contexts,
        projectors=sampler.projectors,
    )
    return [
        step_log_prob(item.recorded, output.velocity)
        for item, output in zip(items, outputs, strict=True)
    ]
