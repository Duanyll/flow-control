"""AWM (Advantage Weighted Matching), arXiv:2509.25050.

:class:`AwmObjective` is the math, :class:`AwmTrainer` the
:class:`~flow_control.training.endpoint.EndpointTrainer` preset. Rollouts are
sampled by the current policy (the lagged EMA when ``objective.off_policy``);
training timesteps are stratified rollout-grid indices over the noisiest 90%
of the plan, skipping the pure-noise step (reference ``discrete_wo_init``).
The lagged EMA is the reference's TRPO-EMA, ``decay = min(0.3, 0.001 * step)``.
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
from .train_timesteps import GridTimesteps, TrainTimesteps

AwmWeighting = Literal["uniform", "t", "t2", "huber", "ghuber"]
AwmKlWeighting = Literal["uniform", "elbo"]


@objective_registry.register("awm")
class AwmObjective(BaseObjective):
    """AWM (Advantage Weighted Matching), arXiv:2509.25050.

    AWM reframes diffusion RL so the policy-gradient objective *is* the
    pretraining flow-matching loss, weighted by the advantage. The
    flow-matching log-likelihood surrogate is::

        log_p(x0) ∝ - w(t) * || v_theta(x_t) - (eps - x0) ||^2

    and the GRPO-style loss is::

        ratio       = exp(log_p - sg[log_p_old])
        policy_loss = mean( max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps)) )
        loss        = policy_loss
                    + beta     * || v_theta - v_ref ||^2   (KL to frozen base)
                    + ema_beta * || v_theta - v_old ||^2   (TRPO-style KL to EMA)

    On-policy (the default), ``log_p_old = sg[log_p]`` so the ratio is 1 in
    value and its gradient reduces to the advantage-weighted flow-matching
    gradient. Off-policy draws ``log_p_old`` (and the rollout endpoints) from
    the lagged EMA policy.
    """

    type: Literal["awm"] = "awm"
    beta: float = 0.001
    """KL-to-reference (frozen base) coefficient."""
    ema_beta: float = 1.0
    """TRPO-style KL-to-EMA coefficient. ``0`` disables the term."""
    adv_clip_max: float = 5.0
    advantage_max: float = 1.0
    """Advantage is clipped to ``[-adv_clip_max, adv_clip_max]`` then rescaled so
    its magnitude is bounded by ``advantage_max``."""
    clip_range: float = 1.0
    """PPO ratio clip epsilon (reference uses 1.0, i.e. effectively unclipped)."""

    weighting: AwmWeighting = "ghuber"
    """Flow-matching log-prob weighting ``w(t)``. ``ghuber`` is the reference
    default; the paper reports ``uniform`` works best."""
    ghuber_power: float = 0.25
    kl_weight: AwmKlWeighting = "uniform"
    kl_ema_weight: AwmKlWeighting = "uniform"

    off_policy: bool = False
    """Sample rollout endpoints and the ratio denominator from the lagged EMA
    policy instead of the current policy."""

    def rollout_policy(self) -> Literal["current", "old"]:
        return "old" if self.off_policy else "current"

    def required_policies(self) -> frozenset[PolicyRole]:
        roles: set[PolicyRole] = set()
        if self.beta > 0:
            roles.add("ref")
        if self.ema_beta > 0 or self.off_policy:
            roles.add("old")
        return frozenset(roles)

    def _flow_matching_logp(
        self,
        velocity: torch.Tensor,
        noise: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted flow-matching log-likelihood surrogate, per sample ``[B]``."""
        target = noise - x0
        mse = ((velocity - target) ** 2).mean(dim=tuple(range(1, velocity.ndim)))
        t_flat = t.view(-1)
        p = self.ghuber_power
        eps = 1e-10
        if self.weighting == "uniform":
            return -mse
        if self.weighting == "t":
            return -mse * t_flat
        if self.weighting == "t2":
            return -mse * t_flat**2
        if self.weighting == "huber":
            return -(torch.sqrt(mse + eps) - 1e-5) * t_flat
        # ghuber
        base = torch.tensor(eps, device=mse.device, dtype=mse.dtype)
        return -(torch.pow(mse + eps, p) - torch.pow(base, p)) * t_flat / p

    def _kl_term(
        self,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        t: torch.Tensor,
        mode: AwmKlWeighting,
    ) -> torch.Tensor:
        diff = ((velocity - reference) ** 2).mean(dim=tuple(range(1, velocity.ndim)))
        if mode == "elbo":
            sigma = t.view(-1)
            std_dev = torch.sqrt(sigma / (1 - torch.clamp(sigma, 0, 0.99))) * 0.7
            diff = diff / (2 * std_dev**2)
        return diff.mean()

    def compute(self, point: TrainPoint, velocities: PolicyVelocities) -> LossOutput:
        forward_prediction = velocities.current
        ref_prediction = self._require(velocities, "ref") if self.beta > 0 else None
        ema_prediction = self._require(velocities, "old") if self.ema_beta > 0 else None

        log_prob = self._flow_matching_logp(
            forward_prediction, point.noise, point.x0, point.t
        )

        if self.off_policy:
            old_log_prob = self._flow_matching_logp(
                self._require(velocities, "old"), point.noise, point.x0, point.t
            )
        else:
            old_log_prob = log_prob.detach()
        ratio = torch.exp(log_prob - old_log_prob.detach())

        adv = point.advantage
        adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        adv = adv / self.adv_clip_max * self.advantage_max
        adv = adv.view(-1)

        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        loss = policy_loss
        kl_loss = torch.tensor(0.0, device=forward_prediction.device)
        ema_kl_loss = torch.tensor(0.0, device=forward_prediction.device)

        if ref_prediction is not None:
            kl_loss = self._kl_term(
                forward_prediction, ref_prediction, point.t, self.kl_weight
            )
            loss = loss + self.beta * kl_loss
        if ema_prediction is not None:
            ema_kl_loss = self._kl_term(
                forward_prediction, ema_prediction, point.t, self.kl_ema_weight
            )
            loss = loss + self.ema_beta * ema_kl_loss

        metrics: dict[str, torch.Tensor] = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "ema_kl_loss": ema_kl_loss.detach(),
            "loss": loss.detach(),
            "ratio_mean": ratio.detach().mean(),
            "clipfrac": (torch.abs(ratio - 1.0) > self.clip_range)
            .float()
            .mean()
            .detach(),
        }
        return LossOutput(loss=loss, metrics=metrics)


@trainer_registry.register("awm")
class AwmTrainer(EndpointTrainer):
    training_type: str = "awm"
    objective: Objective = AwmObjective()
    train_timesteps: TrainTimesteps = GridTimesteps(
        count=6, window=0.9, exclude_first=True, mode="stratified"
    )
    ema_old: EMAConfig = EMAConfig(
        decay=0.3, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.001)
    )
    """TRPO-EMA config (stepped once per epoch)."""


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
    for objective in (
        AwmObjective(),
        AwmObjective(off_policy=True, weighting="uniform"),
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
