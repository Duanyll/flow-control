"""Regenerate the sampler numeric baselines under ``tests/fixtures/``.

The checked-in fixtures were first captured from the pre plan-as-data sampler
(before 164e5a9). ``tests/test_sampler_semantics.py`` replays them against the
current code, so a solver whose numerics drift *accidentally* — a refactor, a
reordered expression, a dependency upgrade — shows up as a bitwise mismatch.

Regenerate only when the algorithm behaviour is changed **on purpose**: run
this script, review what moved, and commit the new fixtures together with the
change that caused them. A fixture diff with no intended algorithm change is a
bug report, not a rebase chore.

Usage: ``uv run python tests/sampler_baseline_capture.py``
"""

from __future__ import annotations

from dataclasses import replace
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
from flow_control.samplers.plan import (
    GuidanceOutput,
    StepContext,
    Transition,
    TransitionResult,
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
from flow_control.samplers.solver.flash import FlashTransition
from flow_control.utils.logging import console

BASELINE_DIR = Path(__file__).resolve().parent / "fixtures/sampler_baselines"

LATENT_SHAPE = (1, 16, 8)

STEP_GRID = torch.linspace(1.0, 0.0, 9).tolist()
"""Sigma grid the per-step sweeps were captured on (8 uniform steps)."""


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


def make_step_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The (latents, velocity, prev_sample) triple every step sweep runs on."""

    def seeded(seed: int) -> torch.Tensor:
        return torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(seed))

    return seeded(21), seeded(22), seeded(23)


def drive_single_eval_transition(
    tr: Transition,
    ctx: StepContext,
    velocity: torch.Tensor,
) -> TransitionResult:
    """Manually run a single-eval run_transition generator with a fixed velocity."""
    gen = tr.run(ctx)
    request = next(gen)
    assert request.sigma == tr.sigma
    try:
        gen.send(GuidanceOutput(velocity=velocity))
    except StopIteration as stop:
        assert isinstance(stop.value, TransitionResult)
        return stop.value
    raise AssertionError("expected a single-eval transition")


def e2e_configs() -> dict[str, Sampler]:
    common: dict[str, Any] = {
        "guidance": ClassifierFreeGuidance(scale=1.0),
        "steps": 8,
    }
    common18: dict[str, Any] = {**common, "steps": 18}
    return {
        "flow_eta0": Sampler(**common, solver=FlowSolver()),
        "ddim_eta0": Sampler(**common, solver=DDIMSolver()),
        "cps_eta0": Sampler(**common, solver=CPSSolver()),
        "dance_eta0": Sampler(**common, solver=DanceSolver()),
        "dpm_order1": Sampler(**common, solver=DPMSolver(order=1)),
        "dpm_order2": Sampler(**common, solver=DPMSolver(order=2)),
        "dpm_order2_steps18": Sampler(**common18, solver=DPMSolver(order=2)),
        "unipc_corrector_on": Sampler(**common, solver=FlowUniPCSolver()),
        "unipc_corrector_off": Sampler(
            **common, solver=FlowUniPCSolver(use_corrector=False)
        ),
        "unipc_steps18": Sampler(**common18, solver=FlowUniPCSolver()),
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


def step_configs() -> dict[str, FlowSolver | DDIMSolver | CPSSolver | DanceSolver]:
    """Stochastic single-eval solvers, swept step by step over ``STEP_GRID``.

    The union is what makes ``step_parts`` visible: it is declared per solver,
    not on ``BaseSolver``, because multi-eval solvers have no such moments.
    """
    return {
        "flow_eta07": FlowSolver(eta=0.7),
        "ddim_eta07": DDIMSolver(eta=0.7),
        "cps_eta07": CPSSolver(eta=0.7),
        "dance_eta07": DanceSolver(eta=0.7),
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
        output = run_phases(model, [phases])[0]
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


def capture_steps() -> None:
    """Per-step moments and replay densities of the stochastic solvers.

    Originally captured through the ``step``/``replay_step`` API deleted in the
    plan-as-data refactor; this reproduces the same fixture layout by driving
    the current ``Transition``/``make_replay`` path, which is exactly what the
    harness asserts against.
    """
    latents, velocity, prev_sample = make_step_tensors()
    for name, solver in step_configs().items():
        entries: list[dict[str, Any]] = []
        # The terminal step lands on sigma_next == 0, where every solver is
        # deterministic regardless of eta; the sweep stops before it.
        for index, (sigma, sigma_next) in enumerate(
            zip(STEP_GRID[:-2], STEP_GRID[1:-1], strict=True)
        ):
            tr = Transition(
                solver=solver,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=solver.eta,
                record=True,
            )
            ctx = StepContext(
                latents=latents,
                generator=torch.Generator().manual_seed(5000 + index),
                solver_state=None,
                guidance_state=None,
            )
            result = drive_single_eval_transition(tr, ctx, velocity)
            assert result.recorded is not None
            mean, std_dev = type(solver).step_parts(
                latents, velocity, sigma, sigma_next, solver.eta
            )[:2]
            replay = solver.make_replay(sigma, sigma_next, solver.eta).logprob(
                velocity, latents, prev_sample
            )
            entries.append(
                {
                    "step_index": index,
                    "sigma": sigma,
                    "sigma_next": sigma_next,
                    "next_latents": result.next_latents,
                    "log_prob": result.recorded.log_prob,
                    "mean": mean,
                    "std_dev": std_dev,
                    "replay_log_prob": replay.log_prob,
                    "replay_mean": replay.mean,
                    "replay_std_dev": replay.std_dev,
                }
            )
        fixture = {
            "solver_config": solver.model_dump(),
            "latents": latents,
            "velocity": velocity,
            "prev_sample": prev_sample,
            "entries": entries,
        }
        torch.save(fixture, BASELINE_DIR / f"step_{name}.pt")
        console.print(f"captured step_{name}: {len(entries)} steps")


def capture_flash_steps() -> None:
    """Flash sweeps the full grid: its terminal step is deterministic anyway.

    Flash has no ``step_parts``; its moments come from ``renoise_parts`` and the
    plan-compiled per-step ``noise_scale``, so it needs its own sweep.
    """
    latents, velocity, prev_sample = make_step_tensors()
    # Non-default scales, and start != end, so the per-step lerp is visible.
    solver = FlashSolver(eta=1.0, noise_scale_start=0.9, noise_scale_end=0.8)
    plan = [
        item for item in solver.plan(STEP_GRID) if isinstance(item, FlashTransition)
    ]
    entries: list[dict[str, Any]] = []
    for index, tr in enumerate(plan):
        ctx = StepContext(
            latents=latents,
            generator=torch.Generator().manual_seed(5000 + index),
            solver_state=None,
            guidance_state=None,
        )
        result = drive_single_eval_transition(replace(tr, record=True), ctx, velocity)
        assert result.recorded is not None
        mean = FlashSolver.renoise_parts(
            latents, velocity, tr.sigma, tr.sigma_next, tr.noise_scale
        )[0]
        replay = result.recorded.replay.logprob(velocity, latents, prev_sample)
        entries.append(
            {
                "step_index": index,
                "sigma": tr.sigma,
                "sigma_next": tr.sigma_next,
                "next_latents": result.next_latents,
                "log_prob": result.recorded.log_prob,
                "mean": mean,
                "std_dev": latents.new_tensor(tr.sigma_next) * tr.noise_scale,
                "replay_log_prob": replay.log_prob,
                "replay_mean": replay.mean,
                "replay_std_dev": replay.std_dev,
            }
        )
    fixture = {
        "solver_config": solver.model_dump(),
        "latents": latents,
        "velocity": velocity,
        "prev_sample": prev_sample,
        "entries": entries,
    }
    torch.save(fixture, BASELINE_DIR / "step_flash_eta1.pt")
    console.print(f"captured step_flash_eta1: {len(entries)} steps")


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
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    capture_e2e()
    capture_steps()
    capture_flash_steps()
    capture_sa_internals()
    console.print(f"baselines written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
