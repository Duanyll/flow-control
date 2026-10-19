"""RAM (Reinforce Adjoint Matching), arXiv:2605.10759.

:class:`RamObjective` is the math, :class:`RamTrainer` the
:class:`~flow_control.training.endpoint.EndpointTrainer` preset. Rollouts are
sampled by the ``old`` variant, which also supplies ``v_old``; training
timesteps are continuous power-law draws, ``K = 8`` per endpoint (reference
``num_loss_targets_per_sample``). The reference draws them on the scheduler's
*shifted* grid, so a faithful config sets ``train_timesteps.weighting.shift``
to the rollout sampler's shift factor (SD3.5: ``3.0``); the preset default
keeps ``shift = 1.0``. Reference RAM does not clip gradients.
"""

from typing import Literal

import torch

from flow_control.samplers import Row

from .endpoint import (
    BaseObjective,
    EndpointTrainer,
    LossOutput,
    Objective,
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

    def rollout_variant(self) -> str:
        return "old"

    def required_variants(self) -> frozenset[str]:
        return frozenset({"old", "ref"})

    def compute(
        self, point: Row, current: torch.Tensor, aux: dict[str, torch.Tensor]
    ) -> LossOutput:
        forward_prediction = current
        base_prediction = aux["ref"].detach()
        old_prediction = aux["old"].detach()
        x0 = point["clean_latents"]

        adv = point["advantage"]
        if self.adv_clip_max is not None:
            adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        scaled_adv = self.reward_multiplier * adv.view(-1, *([1] * (x0.ndim - 1)))

        reward_direction = point["noise"] - x0
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
    clip_grad_norm: float = 0.0


if __name__ == "__main__":
    from rich import print

    torch.manual_seed(0)
    shape = (1, 4, 6, 6)
    x0, noise = torch.randn(shape), torch.randn(shape)
    point: Row = {
        "clean_latents": x0,
        "noise": noise,
        "sigma": 0.7,
        "noisy_latents": 0.3 * x0 + 0.7 * noise,
        "advantage": torch.tensor([1.5]),
    }
    current = torch.randn(shape, requires_grad=True)
    aux = {"old": torch.randn(shape), "ref": torch.randn(shape)}
    out = RamObjective().compute(point, current, aux)
    (grad,) = torch.autograd.grad(out.loss, current)
    assert out.loss.ndim == 0 and out.loss.requires_grad
    assert all(m.ndim == 0 and not m.requires_grad for m in out.metrics.values())
    print(
        f"|grad|={grad.norm().item():.4f}",
        {k: round(v.item(), 4) for k, v in out.metrics.items()},
    )
