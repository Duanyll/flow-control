"""Reward-weighted flow matching (reward-SFT) objective; no trainer preset.

:class:`WeightedFlowMatchingObjective` trains on the current policy's own
rollouts with an advantage-dependent weight and needs no auxiliary policy.
Use it through any :class:`~flow_control.training.endpoint.EndpointTrainer`
preset's ``objective`` block (``"type": "weighted_fm"``).
"""

from typing import Literal

import torch

from .endpoint import (
    BaseObjective,
    LossOutput,
    PolicyRole,
    PolicyVelocities,
    TrainPoint,
    objective_registry,
)


@objective_registry.register("weighted_fm")
class WeightedFlowMatchingObjective(BaseObjective):
    """Reward-weighted flow matching: ``w(A) * || v_theta(x_t) - (eps - x0) ||^2``.

    This is reward-SFT on the current policy's own rollouts. With ``eta < 0``
    it is the Signed RF negative branch (reward-SFT of the same rollout pool
    under ``-eta * R``). No auxiliary policy is needed.
    """

    type: Literal["weighted_fm"] = "weighted_fm"
    eta: float = 1.0
    """Temperature on the advantage; negative values push away from reward."""
    weight: Literal["exp", "linear"] = "exp"
    """``exp``: ``w = exp(eta * A)``; ``linear``: ``w = max(1 + eta * A, 0)``."""

    def rollout_policy(self) -> Literal["current", "old"]:
        return "current"

    def required_policies(self) -> frozenset[PolicyRole]:
        return frozenset()

    def compute(self, point: TrainPoint, velocities: PolicyVelocities) -> LossOutput:
        current = velocities.current
        scaled = self.eta * point.advantage.view(-1)
        if self.weight == "exp":
            weight = torch.exp(scaled)
        else:
            weight = torch.clamp(1.0 + scaled, min=0.0)
        mse = ((current - point.target) ** 2).mean(dim=tuple(range(1, current.ndim)))
        loss = (weight * mse).mean()
        metrics: dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "fm_loss": mse.mean().detach(),
            "weight": weight.mean().detach(),
        }
        return LossOutput(loss=loss, metrics=metrics)


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
    )
    for objective in (
        WeightedFlowMatchingObjective(),
        WeightedFlowMatchingObjective(eta=-1.0, weight="linear"),
    ):
        out = objective.compute(point, velocities)
        (grad,) = torch.autograd.grad(out.loss, velocities.current)
        assert out.loss.ndim == 0 and out.loss.requires_grad
        assert all(m.ndim == 0 and not m.requires_grad for m in out.metrics.values())
        print(objective, f"required={sorted(objective.required_policies())}")
        print(
            f"|grad|={grad.norm().item():.4f}",
            {k: round(v.item(), 4) for k, v in out.metrics.items()},
        )
