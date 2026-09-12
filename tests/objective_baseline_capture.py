"""Freeze the NFT / RAM / AWM objective numerics before the trainer split.

Each trainer carries a per-item loss (``_nft_objective``, ``_ram_objective``,
``_awm_objective``) that the trainer-layering refactor
(``draft/sampler-rethink/09-trainer-layering.md`` §C.1) ports verbatim behind
one ``Objective.compute`` contract. This harness evaluates the *pre-refactor*
losses on fixed inputs and stores loss, gradient and the inputs themselves, so
the port can be proven bitwise instead of "close enough".

Fixture layout, ``tests/fixtures/objective_baselines/{nft,ram,awm}.pt`` — one
flat ``dict[str, Tensor]`` per kind, loadable with ``weights_only=True``:

    <case>/<combo>/loss           0-dim float32 per-item loss
    <case>/<combo>/grad           d loss / d forward, same shape as ``forward``
    <case>/<combo>/x0|noise|forward   latent-shaped float32 inputs
    <case>/<combo>/t|adv          ``[1]`` float32 timestep (1 = pure noise) and
                                  advantage
    <case>/<combo>/old|ref        auxiliary-policy velocities; present only when
                                  the trainer would have computed that role for
                                  the case (AWM's ``ema_prediction`` is ``old``)
    <case>/grad_reaches_old|ref   0-dim bool: whether the pre-refactor code lets
                                  gradient flow into that auxiliary input when it
                                  requires grad (informational; the trainers
                                  compute them under ``no_grad``)

Regenerate only when an objective changes on purpose, and commit the new
fixtures together with the change that caused them.

Usage:
    uv run python tests/objective_baseline_capture.py          # (re)write
    uv run python tests/objective_baseline_capture.py --check  # bitwise replay
"""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import BaseModel, Field

from flow_control.samplers import Sampler, SampleRun
from flow_control.training.awm import AwmTrainer, _AwmLossInput
from flow_control.training.nft import NftTrainer, _NftLossInput
from flow_control.training.ram import RamTrainer, _RamLossInput
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
"""``(case name, kind, trainer field overrides)``; names are unique per kind."""


class _ProbeOverrides(BaseModel):
    """Defaults for the heavy required trainer fields; no model is ever loaded."""

    model: Any = None
    processor: Any = None
    reward: Any = None
    dataset: Any = None
    launch: Any = None
    checkpoint_root: str = ""
    experiment_name: str = "probe"
    seed_checkpoint_dir: str = ""
    num_batches_per_epoch: int = 1
    num_prompts_per_batch: int = 1
    num_rollouts_per_prompt: int = 1
    rollout_sampler: Sampler = Field(default_factory=Sampler)
    validation_sampler: Sampler = Field(default_factory=Sampler)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def log_aggregated_metrics(
        self, metrics: Mapping[str, float | torch.Tensor]
    ) -> None:
        """The objectives log inline; the fixture only keeps loss and grad."""


class _NftProbe(_ProbeOverrides, NftTrainer):
    pass


class _RamProbe(_ProbeOverrides, RamTrainer):
    pass


class _AwmProbe(_ProbeOverrides, AwmTrainer):
    pass


Trainer = NftTrainer | RamTrainer | AwmTrainer

PROBES: dict[str, type[Trainer]] = {
    "nft": _NftProbe,
    "ram": _RamProbe,
    "awm": _AwmProbe,
}


def make_probe(kind: str, kwargs: dict[str, Any]) -> Trainer:
    return PROBES[kind].model_validate({"train_predictor": "model", **kwargs})


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


# ------------------------------------------------------------------------------
# The pre-refactor entry points. S1 replaces ``required_roles`` with
# ``objective.required_policies()`` and ``old_api_loss`` with
# ``objective.compute(point, velocities).loss``; nothing else needs to move.
# ------------------------------------------------------------------------------


def required_roles(trainer: Trainer) -> frozenset[str]:
    """Which auxiliary velocities ``*_loss_batched`` computes for this config."""
    if isinstance(trainer, NftTrainer):
        return frozenset({"old"} | ({"ref"} if trainer.kl_beta > 0 else set()))
    if isinstance(trainer, RamTrainer):
        return frozenset({"old", "ref"})
    return frozenset(
        ({"ref"} if trainer.beta > 0 else set())
        | ({"old"} if trainer._needs_ema_prediction else set())
    )


def old_api_loss(
    trainer: Trainer,
    combo: ComboInputs,
    forward: torch.Tensor,
    old: torch.Tensor | None,
    ref: torch.Tensor | None,
) -> torch.Tensor:
    """Call the per-item objective exactly as the trainer's ``*_loss_batched`` does."""
    if isinstance(trainer, NftTrainer):
        # ``_prepare_nft_loss_input`` interpolates x_t with this exact expression.
        t_expanded = combo.t.view(-1, *([1] * (combo.x0.ndim - 1)))
        xt = (1.0 - t_expanded) * combo.x0 + t_expanded * combo.noise
        prepared = _NftLossInput(
            # Only ``_predict_batched`` reads ``run``; the objective never does.
            run=cast(SampleRun, None),
            timestep=combo.t,
            sigma=float(combo.t.item()),
            x0=combo.x0,
            noisy_latents=xt,
            advantage=combo.adv,
            old_prediction=old,
            ref_prediction=ref,
        )
        return trainer._nft_objective(prepared, forward)
    if isinstance(trainer, RamTrainer):
        return trainer._ram_objective(
            _RamLossInput(
                batch=None,
                x0=combo.x0,
                timestep=combo.t,
                noise=combo.noise,
                advantage=combo.adv,
                base_prediction=ref,
                old_prediction=old,
            ),
            forward,
        )
    return trainer._awm_objective(
        _AwmLossInput(
            batch=None,
            x0=combo.x0,
            timestep=combo.t,
            noise=combo.noise,
            advantage=combo.adv,
            ref_prediction=ref,
            ema_prediction=old,
        ),
        forward,
    )


def evaluate_combo(
    trainer: Trainer, combo: ComboInputs, roles: frozenset[str]
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    """Loss, gradient and the inputs used, plus which aux inputs receive gradient."""
    aux = {role: getattr(combo, role) for role in sorted(roles)}
    forward = combo.forward.clone().requires_grad_(True)
    loss = old_api_loss(trainer, combo, forward, aux.get("old"), aux.get("ref"))
    (grad,) = torch.autograd.grad(loss, forward)

    # Second pass with the aux velocities requiring grad: the trainers never do
    # this (they come from ``no_grad`` forwards), so it must not change loss or
    # the forward gradient; it only reveals whether the objective detaches them.
    leaky = {role: value.clone().requires_grad_(True) for role, value in aux.items()}
    forward_again = combo.forward.clone().requires_grad_(True)
    loss_again = old_api_loss(
        trainer, combo, forward_again, leaky.get("old"), leaky.get("ref")
    )
    grads = torch.autograd.grad(
        loss_again, [forward_again, *leaky.values()], allow_unused=True
    )
    assert torch.equal(loss.detach(), loss_again.detach()) and grads[0] is not None
    assert torch.equal(grad, grads[0])
    reached = {
        role: aux_grad is not None
        for role, aux_grad in zip(leaky, grads[1:], strict=True)
    }

    record = {
        "loss": loss.detach(),
        "grad": grad,
        "x0": combo.x0,
        "noise": combo.noise,
        "t": combo.t,
        "adv": combo.adv,
        "forward": combo.forward,
        **aux,
    }
    return record, reached


def evaluate_case(
    kind: str, kwargs: dict[str, Any], inputs: list[ComboInputs]
) -> dict[str, torch.Tensor]:
    """Flat ``<combo>/<field>`` records for one case, plus the per-case leak flags."""
    trainer = make_probe(kind, kwargs)
    roles = required_roles(trainer)
    out: dict[str, torch.Tensor] = {}
    reached = dict.fromkeys(sorted(roles), False)
    for combo in inputs:
        record, combo_reached = evaluate_combo(trainer, combo, roles)
        for field, value in record.items():
            out[f"{combo.name}/{field}"] = value
        for role, flag in combo_reached.items():
            reached[role] |= flag
    for role, flag in reached.items():
        out[f"grad_reaches_{role}"] = torch.tensor(flag)
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
        mismatched = sorted(
            key
            for key in fixture.keys() | expected.keys()
            if key not in fixture
            or key not in expected
            or not _same(fixture[key], expected[key])
        )
        if mismatched:
            ok = False
            console.print(f"[red]{kind}: {len(mismatched)} mismatched keys[/red]")
            for key in mismatched[:20]:
                console.print(f"  {key}")
        else:
            console.print(f"{kind}: {len(fixture)} tensors match bitwise")
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
