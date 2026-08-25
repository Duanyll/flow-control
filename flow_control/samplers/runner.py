"""Recipe runner: execute per-request built phases.

:func:`run_phases` is the recipe-layer counterpart of ``Sampler.sample``. Each
request brings its OWN built ``list[Phase]`` (recipes build per request so
per-sample generators can pick random windows); every phase is a sequential
barrier, and phase ``i`` of all requests runs in ONE :func:`execute` call so
cross-request batching and FSDP collective alignment are preserved.
Heterogeneous plans within a phase fall back via the executor's grouping.
"""

from __future__ import annotations

import torch

from flow_control.adapters.base import SamplerModel
from flow_control.utils.progress import report_progress

from .executor import Run, execute, validate_distributed_request_count
from .guidance import BaseGuidance
from .plan import RecordedStep, StepContext
from .recipe import FromPrevious, Phase, plan_start_sigma
from .sampler import SampleOutput, make_sample_progress


def _phase_run(
    phase: Phase,
    phase_idx: int,
    previous_latents: torch.Tensor | None,
    guidance: BaseGuidance,
) -> Run:
    if phase.guidance is not guidance and phase.guidance != guidance:
        raise ValueError(
            f"Phase {phase_idx}: all requests must share one guidance config "
            "within a phase (execute() applies a single guidance per call), "
            "but the phases were built with different ones."
        )
    if isinstance(phase.init, FromPrevious):
        if previous_latents is None:
            raise ValueError(
                f"Phase {phase_idx} uses 'from_previous' but there is no "
                "previous phase."
            )
        latents = previous_latents
    else:
        latents = phase.init.build_latents(
            phase.batch, phase.generator, plan_start_sigma(phase.plan)
        )
    return Run(
        plan=phase.plan,
        # Every phase opens a fresh StepContext: solver history never crosses
        # a solver/grid change and guidance state is re-seeded (design:
        # inversion / a new phase gets a new StepContext).
        ctx=StepContext(
            latents=latents,
            generator=phase.generator,
            solver_state=None,
            guidance_state=phase.guidance.init_state(),
        ),
        batch=phase.batch,
        negative_batch=phase.negative_batch,
    )


def run_phases(
    model: SamplerModel,
    requests: list[list[Phase]],
) -> list[SampleOutput]:
    """Execute every request's built phases in cross-request lockstep.

    Returns one :class:`SampleOutput` per request: ``final_latents`` in the
    model dtype, ``timesteps`` = the concatenated executed sigma grid (each
    plan item's start sigma, in execution order, float32 on the model device)
    and ``trajectory`` = the concatenated recorded steps across phases
    (``None`` when nothing recorded).
    """
    if not requests:
        raise ValueError("run_phases requires at least one request.")
    num_phases = len(requests[0])
    if any(len(phases) != num_phases for phases in requests[1:]):
        raise ValueError(
            "All requests must have the same number of phases (they are "
            "built from one shared recipe config); got phase counts "
            f"{sorted({len(phases) for phases in requests})}."
        )
    if num_phases == 0:
        raise ValueError("run_phases requires at least one phase per request.")
    validate_distributed_request_count(len(requests), model.device, "run_phases")

    num_requests = len(requests)
    final_latents: list[torch.Tensor | None] = [None] * num_requests
    executed_sigmas: list[list[float]] = [[] for _ in range(num_requests)]
    recorded: list[list[RecordedStep]] = [[] for _ in range(num_requests)]
    # Per-phase step totals for progress; refreshed from StepEvent when the
    # executor's fallback path reports a larger grouped total.
    totals = [len(phase.plan) for phase in requests[0]]
    completed = 0

    with make_sample_progress() as progress:
        task = progress.add_task("Sampling", total=sum(totals))
        for phase_idx in range(num_phases):
            guidance = requests[0][phase_idx].guidance
            runs = [
                _phase_run(
                    phases[phase_idx], phase_idx, final_latents[request_idx], guidance
                )
                for request_idx, phases in enumerate(requests)
            ]
            for event in execute(model, runs, guidance):
                completed += 1
                totals[phase_idx] = event.total_steps
                total = sum(totals)
                progress.update(task, total=total, advance=1)
                report_progress(completed / total, f"Sampling {completed}/{total}")
            for request_idx, run in enumerate(runs):
                final_latents[request_idx] = run.ctx.latents
                executed_sigmas[request_idx].extend(item.sigma for item in run.plan)
                recorded[request_idx].extend(run.recorded)

    outputs: list[SampleOutput] = []
    for latents, sigmas, steps in zip(
        final_latents, executed_sigmas, recorded, strict=True
    ):
        assert latents is not None
        outputs.append(
            SampleOutput(
                final_latents=latents.to(model.dtype),
                timesteps=torch.tensor(
                    sigmas, dtype=torch.float32, device=model.device
                ),
                trajectory=steps or None,
            )
        )
    return outputs
