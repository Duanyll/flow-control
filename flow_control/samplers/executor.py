"""Rendezvous executor for sampling plans.

Drives every run's transition generators in lockstep: per plan item it collects
the :class:`EvalRequest` each transition yields, merges them into one batched
branch evaluation (:func:`evaluate_branches`, preserving CFG dummy forwards and
FSDP collective alignment), combines the branches through the run's guidance,
and sends the :class:`GuidanceOutput` back. Progress and step-wise consumers
observe :class:`StepEvent` instead of callbacks.

The executor trusts finalized plans: it validates nothing about the config and
keeps only cheap asserts for programming errors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.utils.logging import get_logger, warn_once

from .guidance import BaseGuidance
from .plan import (
    BranchEvals,
    EvalRequest,
    GuidanceOutput,
    RecordedStep,
    SamplingPlan,
    StepContext,
    TransitionResult,
)

logger = get_logger(__name__)

_HETEROGENEOUS_FINGERPRINT = 0


@dataclass(slots=True)
class Run:
    """One request's execution unit: a finalized plan plus its runtime state."""

    plan: SamplingPlan
    ctx: StepContext
    batch: Batch
    negative_batch: Batch | None
    recorded: list[RecordedStep] = field(default_factory=list)


@dataclass(slots=True)
class StepEvent:
    """Emitted after one plan item completes across all runs of a group."""

    step_idx: int
    total_steps: int


@dataclass(slots=True)
class _EvalRound:
    """One batched forward: the (run, request) pairs pending this rendezvous."""

    entries: list[tuple[Run, EvalRequest]]


def validate_distributed_request_count(
    count: int,
    device: torch.device,
    operation: str,
) -> None:
    """Fail fast when ranks would submit unequal request counts to a batched
    sampling entry point (unequal counts desync the FSDP collectives)."""
    if not dist.is_initialized():
        return
    counts = torch.tensor([count, -count], device=device, dtype=torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.MIN)
    if int(counts[0].item()) != -int(counts[1].item()):
        raise ValueError(
            f"All distributed ranks must submit the same number of requests "
            f"to {operation}."
        )


def _sync_negative_pass(negative_pass: bool, device: torch.device) -> bool:
    """Globally agree whether this rendezvous runs a negative pass.

    The all_reduce MAX result is a local variable of the rendezvous — it is
    never written back to any config instance.
    """
    if not dist.is_initialized():
        return negative_pass
    value = torch.tensor(int(negative_pass), device=device, dtype=torch.int64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def evaluate_branches(
    model: BaseModelAdapter,
    guidance: BaseGuidance,
    batches: list[Batch],
    negative_batches: list[Batch | None],
    latents: list[torch.Tensor],
    sigmas: list[float],
    timesteps: list[torch.Tensor],
) -> list[BranchEvals]:
    """Evaluate the cond (and, when needed, uncond) branches for one rendezvous.

    ``timesteps`` are the per-sample tensors fed to the model; ``sigmas`` are
    the same values as plain floats for the :class:`BranchEvals` carriers
    (already known at plan/replay time — no device sync here).

    A sample runs a real negative pass only when the guidance wants one and the
    sample has a negative batch. The decision is synchronized across ranks:
    when the global negative pass is on, samples without a negative batch feed
    their conditional batch as a dummy forward to keep FSDP collectives aligned
    and discard the output (their ``BranchEvals.uncond`` stays ``None``).
    """
    lengths = {
        len(batches),
        len(negative_batches),
        len(latents),
        len(sigmas),
        len(timesteps),
    }
    if lengths == {0}:
        raise ValueError("evaluate_branches requires at least one sample.")
    if len(lengths) != 1:
        raise ValueError(
            "batches, negative_batches, latents, sigmas, and timesteps must "
            "have equal lengths."
        )

    conditional_batches: list[Batch] = []
    for batch, latent in zip(batches, latents, strict=True):
        conditional_batch = batch.copy()
        conditional_batch["noisy_latents"] = latent
        conditional_batches.append(conditional_batch)

    needs_negative = guidance.needs_negative()
    has_negative = [
        needs_negative and negative_batch is not None
        for negative_batch in negative_batches
    ]
    negative_pass = _sync_negative_pass(any(has_negative), model.device)

    conditional = model.predict_velocity_batched(conditional_batches, timesteps)
    if not negative_pass:
        return [
            BranchEvals(cond=cond, uncond=None, latents=latent, sigma=sigma)
            for cond, latent, sigma in zip(conditional, latents, sigmas, strict=True)
        ]

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
            # output is intentionally ignored for samples without a negative.
            unconditional_batches.append(conditional_batch)

    unconditional = model.predict_velocity_batched(unconditional_batches, timesteps)
    return [
        BranchEvals(
            cond=cond,
            uncond=uncond if use_negative else None,
            latents=latent,
            sigma=sigma,
        )
        for cond, uncond, latent, sigma, use_negative in zip(
            conditional, unconditional, latents, sigmas, has_negative, strict=True
        )
    ]


def _plan_fingerprint(plan: SamplingPlan) -> int:
    """Structural fingerprint of a plan's eval topology.

    Built from each item's ``eval_topology()`` key: transition class + solver
    class plus any transition field that changes the eval count (e.g.
    ``SaTransition.final``), so sliced/mixed plans cannot silently desync
    ranks. Sigma values are excluded on purpose: runs may differ in
    resolution-shifted grids while sharing the same transition structure (and
    therefore the same eval topology).
    """
    parts = [item.eval_topology() for item in plan]
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**62) + 1


def _sync_lockstep_choice(fingerprint: int, device: torch.device) -> bool:
    """Single dispatch-time sync; every rank takes the same fast/fallback path."""
    if not dist.is_initialized():
        return fingerprint != _HETEROGENEOUS_FINGERPRINT
    values = torch.tensor([fingerprint, -fingerprint], device=device, dtype=torch.int64)
    dist.all_reduce(values, op=dist.ReduceOp.MIN)
    all_ranks_match = int(values[0].item()) == -int(values[1].item())
    return all_ranks_match and fingerprint != _HETEROGENEOUS_FINGERPRINT


def _sync_eval_target(local_count: int, device: torch.device) -> int:
    """Choose one logical request count for the next fallback rendezvous."""
    if not dist.is_initialized():
        return local_count
    value = torch.tensor(local_count, device=device, dtype=torch.int64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return int(value.item())


def _transition_result(stop: StopIteration) -> TransitionResult:
    result = stop.value
    assert isinstance(result, TransitionResult), (
        "run_transition generators must return a TransitionResult"
    )
    return result


def _apply_result(run: Run, result: TransitionResult, guidance: BaseGuidance) -> None:
    # Contract: solver state is written back only after the transition
    # completes; during execution ctx.solver_state stays the pre-step state.
    run.ctx.latents = result.next_latents
    if result.recorded is not None:
        run.recorded.append(result.recorded)
    if result.reset_runtime_state:
        # Contract rule 2: renoise jumps invalidate live multistep history.
        # Solver state clears to None (plans re-warm from empty history);
        # guidance state re-seeds via THIS execute() call's guidance, matching
        # the initialization at run/phase start (resolved note in the design
        # doc's executor section).
        run.ctx.solver_state = None
        run.ctx.guidance_state = guidance.init_state()
    elif result.next_solver_state is not None:
        run.ctx.solver_state = result.next_solver_state


def _drive(
    runs: list[Run],
    guidance: BaseGuidance,
    first_step_idx: int,
    total_steps: int,
) -> Generator[_EvalRound | StepEvent, list[GuidanceOutput] | None, None]:
    """Lockstep rendezvous over runs sharing one plan topology.

    Yields ``_EvalRound`` whenever a batched eval is needed (the caller sends
    the guided outputs back) and ``StepEvent`` after each completed plan item.
    """
    num_items = len(runs[0].plan)
    assert all(len(run.plan) == num_items for run in runs)
    for item_idx in range(num_items):
        for run in runs:
            # Contract rule 7: guidance state advances per eval, but recorded
            # steps capture the state as of before the transition's first eval.
            run.ctx.pre_transition_guidance_state = run.ctx.guidance_state
        gens = [run.plan[item_idx].run(run.ctx) for run in runs]
        results: list[TransitionResult | None] = [None] * len(runs)
        pending: dict[int, EvalRequest] = {}
        for run_idx, gen in enumerate(gens):
            try:
                pending[run_idx] = next(gen)
            except StopIteration as stop:  # zero-eval transition
                results[run_idx] = _transition_result(stop)
        while pending:
            order = list(pending)
            outputs = yield _EvalRound(
                [(runs[run_idx], pending[run_idx]) for run_idx in order]
            )
            assert outputs is not None and len(outputs) == len(order)
            next_pending: dict[int, EvalRequest] = {}
            for run_idx, output in zip(order, outputs, strict=True):
                try:
                    next_pending[run_idx] = gens[run_idx].send(output)
                except StopIteration as stop:
                    results[run_idx] = _transition_result(stop)
            pending = next_pending
        for run, result in zip(runs, results, strict=True):
            assert result is not None
            _apply_result(run, result, guidance)
        yield StepEvent(step_idx=first_step_idx + item_idx, total_steps=total_steps)


def _eval_round(
    model: BaseModelAdapter,
    guidance: BaseGuidance,
    eval_round: _EvalRound,
    *,
    real_count: int | None = None,
) -> list[GuidanceOutput]:
    if real_count is None:
        real_count = len(eval_round.entries)
    assert 0 <= real_count <= len(eval_round.entries)
    evals = evaluate_branches(
        model=model,
        guidance=guidance,
        batches=[run.batch for run, _ in eval_round.entries],
        # Padding never requests a negative pass on its own. If another rank
        # has a real negative branch, evaluate_branches still supplies the
        # conditional dummy needed to align that global pass.
        negative_batches=[
            run.negative_batch if index < real_count else None
            for index, (run, _) in enumerate(eval_round.entries)
        ],
        latents=[request.latents for _, request in eval_round.entries],
        sigmas=[request.sigma for _, request in eval_round.entries],
        timesteps=[
            request.latents.new_full((1,), request.sigma)
            for _, request in eval_round.entries
        ],
    )
    outputs: list[GuidanceOutput] = []
    for (run, _), branch_evals in zip(
        eval_round.entries[:real_count], evals[:real_count], strict=True
    ):
        # Contract rule 7: the state returned by combine takes effect
        # immediately, including at mid-transition evals.
        output, next_state = guidance.combine(
            branch_evals, run.ctx, run.ctx.guidance_state
        )
        run.ctx.guidance_state = next_state
        outputs.append(output)
    return outputs


def _distributed_rendezvous_eval(
    model: BaseModelAdapter,
    guidance: BaseGuidance,
    eval_round: _EvalRound | None,
    dummy_run: Run,
) -> list[GuidanceOutput] | None:
    """Run one globally aligned fallback eval round.

    Every rank performs exactly one count collective per call. A non-zero
    global target means every rank then submits that many logical samples to
    the adapter; ranks with fewer real requests pad with disposable forwards.
    ``None`` means every rank has drained its local driver.
    """
    local_count = len(eval_round.entries) if eval_round is not None else 0
    target_count = _sync_eval_target(local_count, model.device)
    if target_count == 0:
        return None

    entries = list(eval_round.entries) if eval_round is not None else []
    if local_count < target_count:
        dummy_request = EvalRequest(latents=dummy_run.ctx.latents, sigma=1.0)
        entries.extend(
            (dummy_run, dummy_request) for _ in range(target_count - local_count)
        )
    return _eval_round(
        model,
        guidance,
        _EvalRound(entries),
        real_count=local_count,
    )


def _pump(
    driver: Generator[_EvalRound | StepEvent, list[GuidanceOutput] | None, None],
    eval_fn: Callable[[_EvalRound], list[GuidanceOutput]],
) -> Iterator[StepEvent]:
    try:
        item = next(driver)
        while True:
            if isinstance(item, _EvalRound):
                item = driver.send(eval_fn(item))
            else:
                yield item
                item = next(driver)
    except StopIteration:
        return


def _drive_groups(
    groups: list[list[Run]],
    guidance: BaseGuidance,
) -> Generator[_EvalRound | StepEvent, list[GuidanceOutput] | None, None]:
    """Expose sequential local topology groups as one fallback driver."""
    total_steps = sum(len(group[0].plan) for group in groups)
    offset = 0
    for group in groups:
        yield from _drive(group, guidance, offset, total_steps)
        offset += len(group[0].plan)


def execute(
    model: BaseModelAdapter,
    runs: list[Run],
    guidance: BaseGuidance,
) -> Iterator[StepEvent]:
    """Execute finalized plans for all runs with cross-request batched evals.

    Results accumulate on the :class:`Run` objects (``ctx.latents`` and
    ``recorded``); the returned iterator only reports progress.
    """
    assert runs, "execute requires at least one run"
    device = model.device
    fingerprints = [_plan_fingerprint(run.plan) for run in runs]
    rank_fingerprint = (
        fingerprints[0] if len(set(fingerprints)) == 1 else _HETEROGENEOUS_FINGERPRINT
    )
    if _sync_lockstep_choice(rank_fingerprint, device):
        yield from _pump(
            _drive(runs, guidance, 0, len(runs[0].plan)),
            lambda eval_round: _eval_round(model, guidance, eval_round),
        )
        return

    warn_once(
        logger,
        "Sampling plans have heterogeneous eval topology (within this rank or "
        "across ranks); the executor falls back to grouped sequential execution "
        "with per-round distributed synchronization.",
    )
    grouped_runs: dict[int, list[Run]] = {}
    for run, fingerprint in zip(runs, fingerprints, strict=True):
        grouped_runs.setdefault(fingerprint, []).append(run)

    driver = _drive_groups(list(grouped_runs.values()), guidance)
    try:
        item: _EvalRound | StepEvent | None = next(driver)
    except StopIteration:
        item = None
    while True:
        # Progress events are process-local and require no rank rendezvous.
        while isinstance(item, StepEvent):
            yield item
            try:
                item = next(driver)
            except StopIteration:
                item = None

        eval_round = item if isinstance(item, _EvalRound) else None
        outputs = _distributed_rendezvous_eval(
            model, guidance, eval_round, dummy_run=runs[0]
        )
        if outputs is None:
            return
        if eval_round is None:
            continue
        try:
            item = driver.send(outputs)
        except StopIteration:
            item = None
