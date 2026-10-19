"""Regenerate (or ``--check``) the sampler numeric baselines under ``tests/fixtures/``.

The checked-in fixtures were first captured from the pre plan-as-data sampler
(before 164e5a9). ``tests/test_sampler_semantics.py`` replays them against the
current code, so a solver whose numerics drift *accidentally* — a refactor, a
reordered expression, a dependency upgrade — shows up as a bitwise mismatch.

Regenerate only when the algorithm behaviour is changed **on purpose**: run
this script, review what moved, and commit the new fixtures together with the
change that caused them. A fixture diff with no intended algorithm change is a
bug report, not a rebase chore.

Usage: ``uv run python tests/sampler_baseline_capture.py [--check]``
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from flow_control.samplers import (
    BaseGrid,
    BaseTrajectory,
    BaseTransition,
    Calls,
    Cps,
    Dance,
    Ddim,
    DpmGrid,
    Euler,
    Executor,
    Flash,
    Grid,
    GridFrame,
    Observer,
    RootFrame,
    Row,
    SaGrid,
    StepFrame,
    StepLogProb,
    UniPcGrid,
    data,
    variants,
)
from flow_control.samplers.multistep import sa_update
from flow_control.samplers.shift import ConstantShift, LinearShift
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
    micro_batch_size = 1

    def __init__(self) -> None:
        self.eval_sigmas: list[float] = []
        self.eval_latents: list[torch.Tensor] = []

    def use_variant(self, variant: str | None):
        if variant is not None:
            raise ValueError(f"Test model has no variant {variant!r}.")
        return nullcontext()

    def predict_velocity_batched(
        self,
        batches: list[Row],
        timesteps: list[torch.Tensor],
        *,
        dummy_outputs: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for batch, timestep in zip(batches, timesteps, strict=True):
            latents = batch["noisy_latents"]
            self.eval_latents.append(latents.detach().clone())
            self.eval_sigmas.append(float(timestep.reshape(-1)[0].item()))
            outputs.append(dummy_velocity(latents, timestep))
        return outputs


class ConstVelocityModel(RecordingModel):
    """Duck-typed model returning a fixed velocity for every sample."""

    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.velocity = velocity

    def predict_velocity_batched(self, batches, timesteps, *, dummy_outputs=None):
        super().predict_velocity_batched(
            batches, timesteps, dummy_outputs=dummy_outputs
        )
        return [self.velocity for _ in batches]


def make_row(
    latents: torch.Tensor,
    generator: torch.Generator | None = None,
    observers: Sequence[tuple[type, Observer]] = (),
    **fields: Any,
) -> Row:
    """A sampling row at σ = 1 (what an engine puts on the row before streaming)."""
    return {
        "image_size": (64, 64),
        "clean_latents": torch.zeros_like(latents),
        "noisy_latents": latents.clone(),
        "sigma": 1.0,
        "_generator": generator,
        "_frame": RootFrame(None, observers),
        **fields,
    }


def sample(model: Any, sampler: BaseTrajectory, rows: Iterable[Row]) -> list[Row]:
    """Stream rows through ``sampler``; rows come back in completion order."""
    executor = Executor(model, variants(sampler))
    return list(executor.stream((row, sampler(row)) for row in rows))


def record(row: Row, step: StepFrame) -> None:
    """GRPO-style observer: keep every step's point and likelihood on the row."""
    node = step.node
    if not isinstance(node, BaseTransition) or step.velocity is None:
        return  # a wrapper step (differential_diffusion) predicts nothing itself
    point = {
        "noisy_latents": step.latents,
        "sigma": step.sigma,
        "next_latents": step.next_latents,
        "_frame": step.snapshot(),
    }
    point["log_prob"] = node.log_prob({**data(row), **point}, step.velocity).log_prob
    row.setdefault("trajectory", []).append(point)


def replay(rollout: Row, recorded: Row) -> Calls[StepLogProb]:
    """Re-evaluate one recorded step against the model: GRPO's training path."""
    point = {**data(rollout), **recorded}
    node = point["_frame"].node
    velocity = yield from node.prediction(point)
    return node.log_prob(point, velocity)


def stochastic_points(rollout: Row) -> list[Row]:
    """The recorded steps a GRPO trainer keeps: those with a likelihood."""
    return [
        point
        for point in rollout["trajectory"]
        if point["_frame"].node.is_stochastic(point["_frame"])
    ]


def keep_sigmas(row: Row, grid: GridFrame) -> None:
    row["sigmas"] = grid.sigmas


TRACE_OBSERVERS: list[tuple[type, Observer]] = [
    (BaseTransition, record),
    (BaseGrid, keep_sigmas),
]


def make_initial_latents() -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return torch.randn(LATENT_SHAPE, generator=generator)


def make_step_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The (latents, velocity, prev_sample) triple every step sweep runs on."""

    def seeded(seed: int) -> torch.Tensor:
        return torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(seed))

    return seeded(21), seeded(22), seeded(23)


def e2e_configs() -> dict[str, BaseTrajectory]:
    custom = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.1, 0.0]
    return {
        "flow_eta0": Grid(steps=8),
        "ddim_eta0": Grid(steps=8, transition=Ddim()),
        "cps_eta0": Grid(steps=8, transition=Cps()),
        "dance_eta0": Grid(steps=8, transition=Dance()),
        "dpm_order1": DpmGrid(steps=8, order=1),
        "dpm_order2": DpmGrid(steps=8, order=2),
        "dpm_order2_steps18": DpmGrid(steps=18, order=2),
        "unipc_corrector_on": UniPcGrid(steps=8),
        "unipc_corrector_off": UniPcGrid(steps=8, use_corrector=False),
        "unipc_steps18": UniPcGrid(steps=18),
        "sa_eta0": SaGrid(steps=8, eta=0.0),
        "flash_eta0": Grid(steps=8, transition=Flash(eta=0.0)),
        "flow_eta0_linear_shift": Grid(steps=8, shift=LinearShift()),
        "flow_eta0_diffusers_flow": Grid(
            steps=8, shift=ConstantShift(shift_value=3.0, diffusers_grid=1000)
        ),
        "flow_eta0_custom_sigmas": Grid(steps=8, custom_sigmas=custom),
    }


def step_configs() -> dict[str, Euler | Ddim | Cps | Dance]:
    """Stochastic single-eval transitions, swept step by step over ``STEP_GRID``."""
    return {
        "flow_eta07": Euler(eta=0.7),
        "ddim_eta07": Ddim(eta=0.7),
        "cps_eta07": Cps(eta=0.7),
        "dance_eta07": Dance(eta=0.7),
    }


def e2e_fixture(sampler: BaseTrajectory) -> tuple[dict[str, Any], RecordingModel]:
    """Sample the fixed initial latents through ``sampler`` with a recording model.

    ``traj_latents``/``log_probs`` cover the legacy nonterminal step window of
    transition-based grids; multistep grids pin their path through
    ``eval_latents`` instead.
    """
    model = RecordingModel()
    row = make_row(
        make_initial_latents(), torch.Generator().manual_seed(777), TRACE_OBSERVERS
    )
    (row,) = sample(model, sampler, [row])
    trajectory = row.get("trajectory", [])[:-1]
    return {
        "sampler_config": sampler.model_dump(),
        "initial_latents": make_initial_latents(),
        "final_latents": row["noisy_latents"],
        "traj_latents": torch.stack(
            [trajectory[0]["noisy_latents"]] + [p["next_latents"] for p in trajectory],
            dim=1,
        )
        if trajectory
        else None,
        "log_probs": torch.stack([p["log_prob"] for p in trajectory], dim=1)
        if trajectory
        else None,
        "timesteps": torch.tensor(row["sigmas"][:-1], dtype=torch.float32),
        "eval_sigmas": model.eval_sigmas,
        "eval_latents": model.eval_latents,
    }, model


def drive_step(
    transition: BaseTransition,
    latents: torch.Tensor,
    velocity: torch.Tensor,
    index: int,
    seed: int,
) -> tuple[Row, StepFrame]:
    """Run one transition at position ``index`` of ``STEP_GRID`` with a fixed velocity."""
    frames: list[StepFrame] = []
    row = make_row(
        latents,
        torch.Generator().manual_seed(seed),
        [(BaseTransition, lambda row, step: frames.append(step))],
    )
    grid = GridFrame(Grid(steps=len(STEP_GRID) - 1), row["_frame"])
    grid.sigmas, grid.index = STEP_GRID, index
    row["_frame"], row["sigma"] = grid, STEP_GRID[index]
    Executor(ConstVelocityModel(velocity)).evaluate(
        [transition(row, STEP_GRID[index + 1])]
    )
    return row, frames[0]


def step_fixture(transition: BaseTransition, indices: range) -> dict[str, Any]:
    """Per-step moments and replay densities of one transition over ``STEP_GRID``."""
    latents, velocity, prev_sample = make_step_tensors()
    entries: list[dict[str, Any]] = []
    for index in indices:
        row, step = drive_step(transition, latents, velocity, index, 5000 + index)
        point = {
            "noisy_latents": latents,
            "next_latents": step.next_latents,
            "_frame": step,
        }
        actual = transition.log_prob(point, velocity)
        replay = transition.log_prob({**point, "next_latents": prev_sample}, velocity)
        if isinstance(
            transition, Flash
        ):  # moments of the re-noise form, even when gated off
            mean, std_dev = transition.moments(row, step, latents, velocity)
        else:
            mean, std_dev = actual.mean, actual.std_dev
        entries.append(
            {
                "step_index": index,
                "sigma": step.sigma,
                "sigma_next": step.sigma_next,
                "next_latents": step.next_latents,
                "log_prob": actual.log_prob,
                "mean": mean,
                "std_dev": std_dev,
                "replay_log_prob": replay.log_prob,
                "replay_mean": replay.mean,
                "replay_std_dev": replay.std_dev,
            }
        )
    return {
        "solver_config": transition.model_dump(),
        "latents": latents,
        "velocity": velocity,
        "prev_sample": prev_sample,
        "entries": entries,
    }


def flash_transition() -> Flash:
    # Non-default scales, and start != end, so the per-step lerp is visible.
    return Flash(eta=1.0, noise_scale_start=0.9, noise_scale_end=0.8)


def sa_fixture() -> dict[str, Any]:
    """The SA-Solver's adjusted grid, tau band and AB/AM updates on fixed inputs."""
    solver = SaGrid(steps=8)  # default eta=0.4 > 0
    sigmas = torch.linspace(1.0, 0.0, 9)
    times = sigmas.clone()
    times[0] = solver.initial_time
    times[-2] = times[-3] / 2.0
    x = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(31))
    noise = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(32))
    m0, m1, m2 = (
        torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(seed))
        for seed in (33, 34, 35)
    )
    tau_values = torch.stack([torch.tensor(solver.tau(float(t))) for t in times])
    # Order-1 predictor as used for the first transition of a run.
    ab1 = sa_update(x, tau_values[1], [m0], [times[0]], noise, times[1], 1, False)
    entries = []
    # Order-2 PEC at an in-window time (tau = eta) and outside it (tau = 0).
    for t_index in (3, 6):
        t, tau = times[t_index], tau_values[t_index]
        history = [times[t_index - 2], times[t_index - 1]]
        ab2 = sa_update(x, tau, [m0, m1], history, noise, t, 2, False)
        am2 = sa_update(x, tau, [m0, m1, m2], history, noise, t, 2, True)
        entries.append({"t_index": t_index, "t": t, "tau": tau, "ab2": ab2, "am2": am2})
    return {
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


def fixtures() -> dict[str, dict[str, Any]]:
    """Every fixture by file stem, freshly computed."""
    computed = {f"e2e_{name}": e2e_fixture(s)[0] for name, s in e2e_configs().items()}
    for name, transition in step_configs().items():
        computed[f"step_{name}"] = step_fixture(transition, range(len(STEP_GRID) - 2))
    computed["step_flash_eta1"] = step_fixture(
        flash_transition(), range(len(STEP_GRID) - 1)
    )
    computed["sa_internals"] = sa_fixture()
    return computed


def mismatches(actual: Any, expected: Any, path: str = "") -> list[str]:
    """Paths where two fixture trees differ (tensors bitwise)."""
    if isinstance(expected, dict):
        return [
            m
            for key in expected
            for m in mismatches(actual.get(key), expected[key], f"{path}.{key}")
        ]
    if (
        isinstance(expected, list)
        and expected
        and isinstance(expected[0], dict | torch.Tensor)
    ):
        if len(actual) != len(expected):
            return [f"{path}: {len(actual)} != {len(expected)} entries"]
        return [
            m
            for i, (a, e) in enumerate(zip(actual, expected, strict=True))
            for m in mismatches(a, e, f"{path}[{i}]")
        ]
    if isinstance(expected, torch.Tensor):
        same = isinstance(actual, torch.Tensor) and torch.equal(actual, expected)
        return [] if same else [f"{path}: tensor mismatch"]
    return [] if actual == expected else [f"{path}: {actual!r} != {expected!r}"]


SKIPPED_KEYS = {"sampler_config", "solver_config"}  # config dumps are informational


def check() -> bool:
    ok = True
    for name, computed in fixtures().items():
        stored = torch.load(BASELINE_DIR / f"{name}.pt", weights_only=True)
        stored = {k: v for k, v in stored.items() if k not in SKIPPED_KEYS}
        if (
            name == "e2e_sa_eta0"
        ):  # legacy SA capture: raw grid head, whole-path latents
            stored = {
                k: v
                for k, v in stored.items()
                if k not in ("traj_latents", "log_probs")
            }
            stored["timesteps"] = stored["timesteps"][1:]
            computed = {**computed, "timesteps": computed["timesteps"][1:]}
        if name.startswith("e2e_") and computed["traj_latents"] is None:
            stored.pop("traj_latents", None)
            stored.pop("log_probs", None)
        diff = mismatches(computed, stored, name)
        ok &= not diff
        console.print(
            f"{name}: " + ("[green]ok[/green]" if not diff else f"[red]{diff}[/red]")
        )
    return ok


def main() -> None:
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures().items():
        torch.save(fixture, BASELINE_DIR / f"{name}.pt")
        console.print(f"captured {name}")
    console.print(f"baselines written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
