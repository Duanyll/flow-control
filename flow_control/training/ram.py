"""RAM (Reinforce Adjoint Matching), arXiv:2605.10759.

:class:`RamObjective` is the math, :class:`RamTrainer` the
:class:`~flow_control.training.endpoint.EndpointTrainer` preset. Rollouts are
sampled by the lagged EMA policy, which also supplies ``v_old``; training
timesteps are continuous power-law draws (``p(t) ∝ t``), ``K = 8`` per endpoint
(reference ``num_loss_targets_per_sample``). Reference RAM does not clip
gradients.
"""

from typing import Literal

import torch

from .ema import EMAConfig, LinearRampWarmup
from .endpoint import (
    BaseObjective,
    EndpointTrainer,
    LossOutput,
    Objective,
    PolicyRole,
    PolicyVelocities,
    TrainPoint,
    objective_registry,
)
from .mixins import trainer_registry
from .train_timesteps import ContinuousTimesteps, TrainTimesteps
from .weighting import PowerLawTimestepWeighting


@objective_registry.register("ram")
class RamObjective(BaseObjective):
    """RAM (Reinforce Adjoint Matching), arXiv:2605.10759.

    A trajectory-free objective that regresses the policy velocity onto a
    closed-form target derived from KL-regularized optimal control::

        x_t    = (1 - t) * x0 + t * eps
        target = v_ref(x_t) + m * A * ((eps - x0) - v_old(x_t))
        loss   = || v_theta(x_t) - sg[target] ||^2

    where ``A`` is the group-relative advantage, ``m`` is ``reward_multiplier``,
    ``v_ref`` is the frozen base model and ``v_old`` a lagged EMA of the policy.
    There is no explicit KL term: regularization is implicit through the
    ``v_ref`` anchor and the reward scale. ``|m * A| > 1`` extrapolates past the
    sample velocity by design.
    """

    type: Literal["ram"] = "ram"
    reward_multiplier: float = 100.0
    """The ``m`` in ``target = v_ref + m * A * ((eps - x0) - v_old)``. Doubles as
    the implicit KL knob: larger ``m`` = weaker regularization."""
    adv_clip_max: float | None = None
    """Optional symmetric advantage clip; the reference does not clip."""

    def rollout_policy(self) -> Literal["current", "old"]:
        return "old"

    def required_policies(self) -> frozenset[PolicyRole]:
        return frozenset({"old", "ref"})

    def compute(self, point: TrainPoint, velocities: PolicyVelocities) -> LossOutput:
        forward_prediction = velocities.current
        base_prediction = self._require(velocities, "ref")
        old_prediction = self._require(velocities, "old")

        adv = point.advantage
        if self.adv_clip_max is not None:
            adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        scaled_adv = self.reward_multiplier * adv.view(-1, *([1] * (point.x0.ndim - 1)))

        reward_direction = point.noise - point.x0
        target = base_prediction + scaled_adv * (reward_direction - old_prediction)
        loss = ((forward_prediction - target.detach()) ** 2).mean()

        metrics: dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "target_norm": (target**2).mean().detach(),
            "base_deviate": ((forward_prediction - base_prediction) ** 2)
            .mean()
            .detach(),
        }
        return LossOutput(loss=loss, metrics=metrics)


@trainer_registry.register("ram")
class RamTrainer(EndpointTrainer):
    training_type: str = "ram"
    objective: Objective = RamObjective()
    train_timesteps: TrainTimesteps = ContinuousTimesteps(
        count=8, weighting=PowerLawTimestepWeighting(alpha=1.0)
    )
    ema_old: EMAConfig = EMAConfig(
        decay=0.9, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.01)
    )
    """Old/lagged EMA config (stepped once per epoch): samples endpoints and
    supplies ``v_old`` in the loss target."""
    clip_grad_norm: float = 0.0


if __name__ == "__main__":
    from rich import print

    torch.manual_seed(0)
    shape = (1, 4, 6, 6)
    point = TrainPoint(
        x0=torch.randn(shape),
        noise=torch.randn(shape),
        t=torch.tensor([0.7]),
        advantage=torch.tensor([1.5]),
    )
    velocities = PolicyVelocities(
        current=torch.randn(shape, requires_grad=True),
        old=torch.randn(shape),
        ref=torch.randn(shape),
    )
    for objective in (RamObjective(),):
        out = objective.compute(point, velocities)
        (grad,) = torch.autograd.grad(out.loss, velocities.current)
        assert out.loss.ndim == 0 and out.loss.requires_grad
        assert all(m.ndim == 0 and not m.requires_grad for m in out.metrics.values())
        print(objective, f"required={sorted(objective.required_policies())}")
        print(
            f"|grad|={grad.norm().item():.4f}",
            {k: round(v.item(), 4) for k, v in out.metrics.items()},
        )
