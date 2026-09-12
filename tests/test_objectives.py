"""Objective contract tests: frozen-baseline parity and gradient identities."""

import importlib
import sys
import unittest
from pathlib import Path

import torch

from flow_control.training.objective import (
    AwmObjective,
    BaseObjective,
    NftObjective,
    PolicyVelocities,
    RamObjective,
    TrainPoint,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
capture = importlib.import_module("objective_baseline_capture")


def _grad(
    objective: BaseObjective, point: TrainPoint, velocities: PolicyVelocities
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss and d loss / d current for one compute call."""
    out = objective.compute(point, velocities)
    (grad,) = torch.autograd.grad(out.loss, velocities.current)
    return out.loss.detach(), grad


class ObjectiveParityTest(unittest.TestCase):
    def test_objectives_match_frozen_baselines(self) -> None:
        """Pins the 2026-09 nft/ram/awm -> Objective split bitwise.

        The fixtures in ``tests/fixtures/objective_baselines`` were captured
        from the trainer-era ``_nft_objective`` / ``_ram_objective`` /
        ``_awm_objective`` (commit 97a84b3). Every case must reproduce loss and
        d loss/d current with ``torch.equal``, the required roles must match
        the roles the old ``*_loss_batched`` gating computed, and ``old``/``ref``
        must receive no gradient (the trainer-era code leaked into them; the
        contract detaches).
        """
        for kind in capture.KINDS:
            fixture: dict[str, torch.Tensor] = torch.load(
                capture.BASELINE_DIR / f"{kind}.pt", weights_only=True
            )
            for case, kwargs in capture.cases_of(kind):
                objective = capture.make_objective(kind, kwargs)
                inputs = capture.inputs_from_fixture(fixture, case)
                self.assertTrue(inputs, f"{kind}/{case}: no combos in fixture")
                for combo in inputs:
                    with self.subTest(kind=kind, case=case, combo=combo.name):
                        stored = {
                            role
                            for role in ("old", "ref")
                            if getattr(combo, role) is not None
                        }
                        self.assertEqual(set(objective.required_policies()), stored)

                        current = combo.forward.clone().requires_grad_(True)
                        aux = {
                            role: getattr(combo, role).clone().requires_grad_(True)
                            for role in sorted(stored)
                        }
                        point = TrainPoint(
                            x0=combo.x0,
                            noise=combo.noise,
                            t=combo.t,
                            advantage=combo.adv,
                        )
                        out = objective.compute(
                            point,
                            PolicyVelocities(current, aux.get("old"), aux.get("ref")),
                        )
                        grads = torch.autograd.grad(
                            out.loss, [current, *aux.values()], allow_unused=True
                        )
                        prefix = f"{case}/{combo.name}"
                        self.assertTrue(
                            torch.equal(out.loss.detach(), fixture[f"{prefix}/loss"])
                        )
                        self.assertIsNotNone(grads[0])
                        self.assertTrue(
                            torch.equal(grads[0], fixture[f"{prefix}/grad"])
                        )
                        for role, aux_grad in zip(aux, grads[1:], strict=True):
                            self.assertIsNone(aux_grad, f"gradient leaked into {role}")
                        for name, value in out.metrics.items():
                            self.assertEqual(value.ndim, 0, name)
                            self.assertFalse(value.requires_grad, name)


class ObjectiveGradientIdentityTest(unittest.TestCase):
    def test_objective_gradient_identities(self) -> None:
        """Pins the mapping in draft/sampler-rethink/01-math-landscape.md §B.2.

        Each verbatim objective's gradient w.r.t. the current velocity equals a
        closed-form affine regression ``2W/N * (v - v*)`` in float64: RAM is the
        signed target, on-policy uniform AWM is advantage-weighted FM plus the
        two KL pulls, and NFT (kl_beta=0) collapses its two branches into one
        target ``c_old * v_old + c_sample * (noise - x0)``.
        """
        torch.manual_seed(0)
        shape = (1, 3, 4, 5)
        n = 1
        for dim in shape:
            n *= dim
        rtol = 1e-9

        def randn() -> torch.Tensor:
            return torch.randn(shape, dtype=torch.float64)

        for t_value, adv_value in ((0.05, -3.0), (0.5, 0.7), (0.95, 8.0)):
            x0, noise, v_old, v_ref = randn(), randn(), randn(), randn()
            v = randn().requires_grad_(True)
            t = torch.tensor([t_value], dtype=torch.float64)
            adv = torch.tensor([adv_value], dtype=torch.float64)
            point = TrainPoint(x0=x0, noise=noise, t=t, advantage=adv)
            target = noise - x0
            velocities = PolicyVelocities(current=v, old=v_old, ref=v_ref)

            with self.subTest(objective="ram", t=t_value, adv=adv_value):
                m = 100.0
                _, grad = _grad(RamObjective(reward_multiplier=m), point, velocities)
                target_ram = v_ref + m * adv * (target - v_old)
                torch.testing.assert_close(
                    grad, 2.0 / n * (v - target_ram), rtol=rtol, atol=0
                )

            with self.subTest(objective="awm", t=t_value, adv=adv_value):
                beta, ema_beta, clip, adv_max = 0.01, 0.5, 5.0, 1.0
                objective = AwmObjective(
                    beta=beta,
                    ema_beta=ema_beta,
                    adv_clip_max=clip,
                    advantage_max=adv_max,
                    clip_range=1.0,
                    weighting="uniform",
                )
                _, grad = _grad(objective, point, velocities)
                adv_scaled = adv.clamp(-clip, clip) / clip * adv_max
                expected = (
                    adv_scaled * 2.0 / n * (v - target)
                    + beta * 2.0 / n * (v - v_ref)
                    + ema_beta * 2.0 / n * (v - v_old)
                )
                torch.testing.assert_close(grad, expected, rtol=rtol, atol=0)

            with self.subTest(objective="nft", t=t_value, adv=adv_value):
                beta, c = 0.7, 5.0
                objective = NftObjective(
                    beta=beta, kl_beta=0.0, adv_clip_max=c, adv_mode="all"
                )
                _, grad = _grad(
                    objective, point, PolicyVelocities(current=v, old=v_old)
                )
                with torch.no_grad():
                    v_pos = beta * v + (1 - beta) * v_old
                    v_neg = (1 + beta) * v_old - beta * v
                    w_pos = (t * (target - v_pos)).abs().mean().clip(min=1e-5)
                    w_neg = (t * (target - v_neg)).abs().mean().clip(min=1e-5)
                    r = ((adv.clamp(-c, c) / c) / 2 + 0.5).clamp(0.0, 1.0)
                    mix = r / w_pos + (1 - r) / w_neg
                    c_sample = (r / w_pos - (1 - r) / w_neg) / (beta * mix)
                    c_old = 1 - c_sample
                    weight = c * beta * t**2 * mix
                    v_star = c_old * v_old + c_sample * target
                    expected = 2 * weight / n * (v - v_star)
                torch.testing.assert_close(grad, expected, rtol=rtol, atol=0)


if __name__ == "__main__":
    unittest.main()
