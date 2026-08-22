"""Capture numeric baselines of the pre-refactor sampler implementation.

The fixtures under ``draft/sampler_migration/baselines/`` were captured from
the OLD (pre plan-as-data) implementation and are replayed by
``tests/test_sampler_semantics.py`` against the refactored code. Fixtures are
gitignored (``draft/``).

The legacy ``step``/``replay_step`` API was deleted in Phase 1b, the
stacked-trajectory ``SampleOutput`` fields in Phase 2 and
``sample(return_trajectory=True)`` in Phase 5; the per-step ``step_*.pt``
fixtures can only be regenerated from the pre-refactor revision of this script
(``git log`` for "baseline capture script"). ``capture_e2e`` still runs on the
current tree — it records the full window through the recipe runner and
reconstructs the legacy stacked fixture layout from ``SampleOutput.trajectory``
(note: for SA the recaptured ``timesteps`` are the plan-compiled grid, whose
adjusted head/penultimate entries the harness skips) — and the harness imports
the shared fixture-path/model/batch helpers from here.

Usage: ``uv run python tests/sampler_baseline_capture.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from flow_control.adapters.base import Batch
from flow_control.samplers import (
    ClassifierFreeGuidance,
    PhaseConfig,
    PhasesRecipe,
    RecipeBuildContext,
    SdeWindow,
    run_phases,
)
from flow_control.samplers.sampler import Sampler
from flow_control.samplers.shift import ConstantShift, LinearShift
from flow_control.samplers.solver import (
    CPSSolver,
    DanceSolver,
    DDIMSolver,
    DPMSolver,
    FlashSolver,
    FlowSolver,
    FlowUniPCSolver,
    SASolver,
)
from flow_control.utils.logging import console

BASELINE_DIR = Path(__file__).resolve().parent.parent / (
    "draft/sampler_migration/baselines"
)

LATENT_SHAPE = (1, 16, 8)


def dummy_velocity(latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Deterministic velocity: nonlinear in latents, dependent on sigma."""
    sigma = timestep.reshape(-1)[0]
    return torch.tanh(latents) * 0.7 + (sigma - 0.5) * 0.3 * torch.ones_like(latents)


class RecordingModel:
    """Duck-typed model that logs every (latents, sigma) evaluation."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.eval_sigmas: list[float] = []
        self.eval_latents: list[torch.Tensor] = []

    def predict_velocity_batched(
        self,
        batches: list[Batch],
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for batch, timestep in zip(batches, timesteps, strict=True):
            latents = batch["noisy_latents"]
            self.eval_latents.append(latents.detach().clone())
            self.eval_sigmas.append(float(timestep.reshape(-1)[0].item()))
            outputs.append(dummy_velocity(latents, timestep))
        return outputs


def make_request_batch(latents: torch.Tensor) -> Batch:
    return {
        "image_size": (64, 64),
        "clean_latents": torch.zeros_like(latents),
        "noisy_latents": latents.clone(),
    }


def make_initial_latents() -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return torch.randn(LATENT_SHAPE, generator=generator)


def e2e_configs() -> dict[str, Sampler]:
    common: dict[str, Any] = {
        "guidance": ClassifierFreeGuidance(scale=1.0),
        "steps": 8,
    }
    return {
        "flow_eta0": Sampler(**common, solver=FlowSolver()),
        "ddim_eta0": Sampler(**common, solver=DDIMSolver()),
        "cps_eta0": Sampler(**common, solver=CPSSolver()),
        "dance_eta0": Sampler(**common, solver=DanceSolver()),
        "dpm_order1": Sampler(**common, solver=DPMSolver(order=1)),
        "dpm_order2": Sampler(**common, solver=DPMSolver(order=2)),
        "dpm_order2_steps18": Sampler(
            **{**common, "steps": 18}, solver=DPMSolver(order=2)
        ),
        "unipc_corrector_on": Sampler(**common, solver=FlowUniPCSolver()),
        "unipc_corrector_off": Sampler(
            **common, solver=FlowUniPCSolver(use_corrector=False)
        ),
        "unipc_steps18": Sampler(**{**common, "steps": 18}, solver=FlowUniPCSolver()),
        "sa_eta0": Sampler(**common, solver=SASolver(eta=0.0)),
        "flash_eta0": Sampler(**common, solver=FlashSolver(eta=0.0)),
        "flow_eta0_linear_shift": Sampler(
            **common, solver=FlowSolver(), shift=LinearShift()
        ),
        "flow_eta0_diffusers_flow": Sampler(
            **common,
            solver=FlowSolver(),
            sigma_schedule="diffusers_flow",
            shift=ConstantShift(shift_value=3.0),
        ),
        "flow_eta0_custom_sigmas": Sampler(
            **common,
            solver=FlowSolver(),
            custom_sigmas=[1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.1, 0.0],
        ),
    }


def capture_e2e() -> None:
    initial_latents = make_initial_latents()
    recipe = PhasesRecipe(phases=[PhaseConfig(transforms=[SdeWindow(record=True)])])
    for name, sampler in e2e_configs().items():
        model = RecordingModel()
        phases = recipe.build(
            RecipeBuildContext(
                default_sampler=sampler,
                batches={"main": make_request_batch(initial_latents)},
                negative_batch_for=lambda name: None,
                generator=torch.Generator().manual_seed(777),
            )
        )
        output = run_phases(model, [phases])[0]  # type: ignore[arg-type]
        # Reconstruct the legacy stacked fixture layout from the recorded
        # trajectory (with the default full window this matches the old
        # SampleOutput.latents / log_probs / timesteps fields bitwise).
        trajectory = output.trajectory or []
        fixture = {
            "sampler_config": sampler.model_dump(),
            "initial_latents": initial_latents,
            "final_latents": output.final_latents,
            "traj_latents": torch.stack(
                [trajectory[0].latent_t] + [step.latent_next for step in trajectory],
                dim=1,
            ),
            "log_probs": torch.stack([step.log_prob for step in trajectory], dim=1),
            "timesteps": output.timesteps,
            "eval_sigmas": model.eval_sigmas,
            "eval_latents": model.eval_latents,
        }
        torch.save(fixture, BASELINE_DIR / f"e2e_{name}.pt")
        console.print(f"captured e2e_{name}: {len(model.eval_sigmas)} evals")


def capture_sa_internals() -> None:
    solver = SASolver()  # default eta=0.4 > 0
    sigmas = torch.linspace(1.0, 0.0, 9)
    times = sigmas.clone()
    times[0] = solver.initial_time
    times[-2] = times[-3] / 2.0

    x = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(31))
    noise = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(32))
    m0 = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(33))
    m1 = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(34))
    m2 = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(35))

    tau_values = torch.stack([torch.tensor(solver._tau(float(t))) for t in times])

    # Order-1 predictor as used for the first transition of a run.
    ab1 = solver._adams_bashforth_update(
        x,
        torch.tensor(solver._tau(float(times[1]))),
        [m0],
        [times[0]],
        noise,
        times[1],
        order=1,
    )

    entries = []
    # Order-2 PEC at an in-window time (tau = eta) and outside it (tau = 0).
    for t_index in (3, 6):
        t = times[t_index]
        tau = torch.tensor(solver._tau(float(t)))
        time_history = [times[t_index - 2], times[t_index - 1]]
        ab2 = solver._adams_bashforth_update(
            x, tau, [m0, m1], time_history, noise, t, order=2
        )
        am2 = solver._adams_moulton_update(
            x, tau, [m0, m1, m2], time_history, noise, t, order=2
        )
        entries.append({"t_index": t_index, "t": t, "tau": tau, "ab2": ab2, "am2": am2})

    fixture = {
        "solver_config": solver.model_dump(),
        "sigmas": sigmas,
        "times": times,
        "tau_values": tau_values,
        "x": x,
        "noise": noise,
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "ab1": ab1,
        "entries": entries,
    }
    torch.save(fixture, BASELINE_DIR / "sa_internals.pt")
    console.print("captured sa_internals")


def main() -> None:
    # step_*.pt fixtures are regenerable only from the pre-refactor revision
    # of this script (they pinned the deleted step/replay_step API).
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    capture_e2e()
    capture_sa_internals()
    console.print(f"baselines written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
