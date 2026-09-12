"""Freeze the NFT / RAM / AWM objective numerics across the trainer split.

The fixtures were captured from the pre-refactor per-item losses
(``_nft_objective``, ``_ram_objective``, ``_awm_objective``) that the
trainer-layering refactor (``draft/sampler-rethink/09-trainer-layering.md``
§C.1) ported verbatim behind one ``Objective.compute`` contract
(``flow_control.training.objective``). This harness evaluates the objectives on
fixed inputs and stores loss, gradient and the inputs themselves, so the port
is proven bitwise instead of "close enough"; ``tests/test_objectives.py``
replays the same fixtures as a collected test.

Fixture layout, ``tests/fixtures/objective_baselines/{nft,ram,awm}.pt`` — one
flat ``dict[str, Tensor]`` per kind, loadable with ``weights_only=True``:

    <case>/<combo>/loss           0-dim float32 per-item loss
    <case>/<combo>/grad           d loss / d forward, same shape as ``forward``
    <case>/<combo>/x0|noise|forward   latent-shaped float32 inputs
    <case>/<combo>/t|adv          ``[1]`` float32 timestep (1 = pure noise) and
                                  advantage
    <case>/<combo>/old|ref        auxiliary-policy velocities; present only when
                                  ``objective.required_policies()`` contains the
                                  role (AWM's ``ema_prediction`` is ``old``)
    <case>/grad_reaches_old|ref   0-dim bool, pre-refactor record only: whether
                                  the trainer-era code let gradient reach that
                                  auxiliary input. ``Objective.compute`` detaches
                                  ``old``/``ref`` by contract, so ``--check``
                                  skips these keys and a recapture omits them.

Regenerate only when an objective changes on purpose, and commit the new
fixtures together with the change that caused them.

Usage:
    uv run python tests/objective_baseline_capture.py          # (re)write
    uv run python tests/objective_baseline_capture.py --check  # bitwise replay
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import TypeAdapter

from flow_control.training.objective import (
    BaseObjective,
    Objective,
    PolicyVelocities,
    TrainPoint,
)
from flow_control.utils.logging import console

BASELINE_DIR = Path(__file__).resolve().parent / "fixtures/objective_baselines"

KINDS = ("nft", "ram", "awm")
SHAPES = ((1, 4, 6, 6), (1, 2, 3, 5))
"""A conventional BCHW latent plus an odd-sized one to catch dim assumptions."""
TIMESTEPS = (0.05, 0.5, 0.95)
ADVANTAGES = (-3.0, -0.5, 0.0, 0.7, 8.0)
"""Spans both clip sides of NFT/AWM (|A| > 5), a zero, and small values."""

CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("default", "nft", {}),
    ("kl_beta0", "nft", {"kl_beta": 0.0}),
    ("positive_only", "nft", {"adv_mode": "positive_only"}),
    ("negative_only", "nft", {"adv_mode": "negative_only"}),
    ("binary", "nft", {"adv_mode": "binary"}),
    ("beta05", "nft", {"beta": 0.5}),
    ("default", "ram", {}),
    ("adv_clip1", "ram", {"adv_clip_max": 1.0}),
    ("multiplier1000", "ram", {"reward_multiplier": 1000.0}),
    ("default", "awm", {}),
    ("uniform", "awm", {"weighting": "uniform"}),
    ("t", "awm", {"weighting": "t"}),
    ("t2", "awm", {"weighting": "t2"}),
    ("huber", "awm", {"weighting": "huber"}),
    ("off_policy", "awm", {"off_policy": True}),
    ("no_kl", "awm", {"beta": 0.0, "ema_beta": 0.0}),
    ("elbo", "awm", {"kl_weight": "elbo", "kl_ema_weight": "elbo"}),
    ("advantage_max05", "awm", {"advantage_max": 0.5}),
]
"""``(case name, kind, objective field overrides)``; names are unique per kind."""


_objective_adapter = TypeAdapter(Objective)


def make_objective(kind: str, kwargs: dict[str, Any]) -> BaseObjective:
    return _objective_adapter.validate_python({"type": kind, **kwargs})


@dataclass(slots=True)
class ComboInputs:
    """One (shape, t, advantage) evaluation point; ``old``/``ref`` may be unused."""

    name: str
    x0: torch.Tensor
    noise: torch.Tensor
    t: torch.Tensor
    adv: torch.Tensor
    old: torch.Tensor | None
    ref: torch.Tensor | None
    forward: torch.Tensor


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def make_inputs(seed: int = 0) -> list[ComboInputs]:
    """Deterministic inputs shared by every case (the fixture stores them anyway)."""
    generator = torch.Generator().manual_seed(seed)
    combos: list[ComboInputs] = []
    for shape_index, shape in enumerate(SHAPES):
        for t in TIMESTEPS:
            for adv in ADVANTAGES:
                combos.append(
                    ComboInputs(
                        name=f"s{shape_index}_t{t}_a{adv}",
                        x0=_randn(shape, generator),
                        noise=_randn(shape, generator),
                        old=_randn(shape, generator),
                        ref=_randn(shape, generator),
                        forward=_randn(shape, generator),
                        t=torch.tensor([t], dtype=torch.float32),
                        adv=torch.tensor([adv], dtype=torch.float32),
                    )
                )
    return combos


def inputs_from_fixture(
    fixture: dict[str, torch.Tensor], case: str
) -> list[ComboInputs]:
    """Rebuild the combos of one case from a saved fixture (no RNG involved)."""
    prefix = f"{case}/"
    combo_names = [
        key[len(prefix) : -len("/x0")]
        for key in fixture
        if key.startswith(prefix) and key.endswith("/x0")
    ]
    return [
        ComboInputs(
            name=name,
            x0=fixture[f"{prefix}{name}/x0"],
            noise=fixture[f"{prefix}{name}/noise"],
            t=fixture[f"{prefix}{name}/t"],
            adv=fixture[f"{prefix}{name}/adv"],
            old=fixture.get(f"{prefix}{name}/old"),
            ref=fixture.get(f"{prefix}{name}/ref"),
            forward=fixture[f"{prefix}{name}/forward"],
        )
        for name in combo_names
    ]


def evaluate_combo(
    objective: BaseObjective, combo: ComboInputs, roles: frozenset[str]
) -> dict[str, torch.Tensor]:
    """Loss, gradient and the inputs used for one combo."""
    aux = {role: getattr(combo, role) for role in sorted(roles)}
    forward = combo.forward.clone().requires_grad_(True)
    point = TrainPoint(
        x0=combo.x0, noise=combo.noise, t=combo.t, advantage=combo.adv, grid_index=None
    )
    loss = objective.compute(
        point, PolicyVelocities(forward, aux.get("old"), aux.get("ref"))
    ).loss
    (grad,) = torch.autograd.grad(loss, forward)
    return {
        "loss": loss.detach(),
        "grad": grad,
        "x0": combo.x0,
        "noise": combo.noise,
        "t": combo.t,
        "adv": combo.adv,
        "forward": combo.forward,
        **aux,
    }


def evaluate_case(
    kind: str, kwargs: dict[str, Any], inputs: list[ComboInputs]
) -> dict[str, torch.Tensor]:
    """Flat ``<combo>/<field>`` records for one case."""
    objective = make_objective(kind, kwargs)
    roles = frozenset(objective.required_policies())
    out: dict[str, torch.Tensor] = {}
    for combo in inputs:
        for field, value in evaluate_combo(objective, combo, roles).items():
            out[f"{combo.name}/{field}"] = value
    return out


def cases_of(kind: str) -> list[tuple[str, dict[str, Any]]]:
    return [(name, kwargs) for name, case_kind, kwargs in CASES if case_kind == kind]


def capture() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs()
    for kind in KINDS:
        fixture: dict[str, torch.Tensor] = {}
        for name, kwargs in cases_of(kind):
            for key, value in evaluate_case(kind, kwargs, inputs).items():
                fixture[f"{name}/{key}"] = value
        path = BASELINE_DIR / f"{kind}.pt"
        torch.save(fixture, path)
        console.print(
            f"captured {kind}: {len(cases_of(kind))} cases, {len(fixture)} tensors, "
            f"{path.stat().st_size / 1024:.0f} KiB"
        )


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def check() -> bool:
    """Recompute from the saved inputs and compare bitwise; True when all match."""
    ok = True
    for kind in KINDS:
        fixture: dict[str, torch.Tensor] = torch.load(
            BASELINE_DIR / f"{kind}.pt", weights_only=True
        )
        expected: dict[str, torch.Tensor] = {}
        for name, kwargs in cases_of(kind):
            inputs = inputs_from_fixture(fixture, name)
            for key, value in evaluate_case(kind, kwargs, inputs).items():
                expected[f"{name}/{key}"] = value
        # ``grad_reaches_*`` flags are the pre-refactor leak record (see the
        # module docstring); the objectives detach by contract, so skip them.
        mismatched = sorted(
            key
            for key in fixture.keys() | expected.keys()
            if "/grad_reaches_" not in key
            and (
                key not in fixture
                or key not in expected
                or not _same(fixture[key], expected[key])
            )
        )
        if mismatched:
            ok = False
            console.print(f"[red]{kind}: {len(mismatched)} mismatched keys[/red]")
            for key in mismatched[:20]:
                console.print(f"  {key}")
        else:
            console.print(f"{kind}: {len(expected)} tensors match bitwise")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="replay the saved fixtures instead of rewriting them",
    )
    args = parser.parse_args()
    if args.check:
        if not check():
            raise SystemExit(1)
    else:
        capture()
        console.print(f"baselines written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
