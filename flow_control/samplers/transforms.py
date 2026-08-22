"""Pure plan-to-plan transforms (axis 1 of the plan-as-data design).

Transforms are ordinary functions over :data:`SamplingPlan` lists; they never
touch models or latents. The one RNG consumer here is :func:`select_sde_window`,
which *selects* a window rather than transforming a plan. ``finalize_replay_state``
is the idempotent final step of any plan pipeline — call it after the last
transform, before handing the plan to the executor. The executor trusts
finalized plans and re-checks nothing.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .plan import RenoiseTransition, SamplingPlan


def select_sde_window(
    num_transitions: int,
    size: int | None,
    window_range: tuple[int, int] | None,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    """Pick the ``[start, end)`` denoise-index window for :func:`with_sde_window`.

    ``size=None`` is the full window ``(0, num_transitions - 1)`` — the terminal
    transition is always excluded. Otherwise the start is drawn uniformly from
    ``generator`` so the window fits inside ``window_range`` (which defaults to
    the full window).
    """
    if size is None:
        return 0, num_transitions - 1

    if size <= 0:
        raise ValueError("SDE window size must be positive.")
    range_start, range_end = window_range or (0, num_transitions - 1)
    if not 0 <= range_start < range_end < num_transitions:
        raise ValueError(
            "SDE window range must satisfy 0 <= start < end < num_transitions."
        )
    max_start = range_end - size
    if max_start < range_start:
        raise ValueError(
            f"SDE window size={size} does not fit in "
            f"range=({range_start}, {range_end})."
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
    return window_start, window_start + size


def with_sde_window(
    plan: SamplingPlan,
    start: int,
    end: int,
    *,
    record: bool = False,
) -> SamplingPlan:
    """Gate per-step eta to the denoise-index window ``[start, end)``.

    Outside the window transitions become deterministic (``eta=0``). With
    ``record=True`` the same window is marked as the replayable trajectory.
    Window indices count denoise transitions only; control transitions
    (:class:`RenoiseTransition`) pass through unchanged.
    """
    result: SamplingPlan = []
    denoise_index = 0
    for item in plan:
        if isinstance(item, RenoiseTransition):
            result.append(item)
            continue
        active = start <= denoise_index < end
        if (
            record
            and active
            and item.eta > 0
            and not item.solver.supports_step_log_prob
        ):
            raise NotImplementedError(f"{item.solver.type} has no step log-prob replay")
        result.append(
            replace(
                item,
                eta=item.eta if active else 0.0,
                record=item.record or (record and active),
            )
        )
        denoise_index += 1
    return result


def invert_first_order(plan: SamplingPlan) -> SamplingPlan:
    """Reverse a first-order deterministic plan (approximate DDIM inversion).

    Each transition is mirrored (``sigma`` and ``sigma_next`` swapped), the
    order reversed, and ``eta`` forced to 0 — the shared deterministic Euler
    path run backwards. Only meaningful for stateless first-order solvers whose
    ``eta=0`` update is symmetric under the sigma swap; those expose it via
    ``BaseSolver.invert``. Note Euler inversion is the standard approximation:
    reconstruction is not exact for state-dependent velocity fields.
    """
    result: SamplingPlan = []
    for item in reversed(plan):
        if isinstance(item, RenoiseTransition):
            raise NotImplementedError(
                "Cannot invert a plan containing control transitions "
                "(RenoiseTransition)."
            )
        result.append(
            replace(
                item,
                sigma=item.sigma_next,
                sigma_next=item.sigma,
                eta=0.0,
                record=False,
                save_solver_state=False,
            )
        )
    return result


def finalize_replay_state(plan: SamplingPlan) -> SamplingPlan:
    """Decide which recorded transitions must snapshot pre-step solver state.

    Idempotent; must run as the last step of a plan pipeline so solvers see the
    full plan (including control transitions) when queried.
    """
    result: SamplingPlan = []
    for plan_item_index, item in enumerate(plan):
        if isinstance(item, RenoiseTransition):
            result.append(item)
            continue
        result.append(
            replace(
                item,
                save_solver_state=(
                    item.record
                    and item.solver.requires_replay_state(plan, plan_item_index)
                ),
            )
        )
    return result
