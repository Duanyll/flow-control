"""Per-item training objectives for endpoint-based diffusion RL.

An :class:`Objective` is the math core of a rollout trainer: given one training
point (a clean endpoint re-noised at a timestep) and the velocities the
candidate policies predict there, it returns a loss and detached metrics. It
holds no device, model or logger; the trainer owns the loop, decides which
policy forwards to run (:meth:`BaseObjective.required_policies`), which policy
samples the rollouts (:meth:`BaseObjective.rollout_policy`), and prefixes the
metrics with ``train/`` before aggregation.

Conventions (flow_control): ``t`` in ``[0, 1]`` with ``1`` = pure noise,
``x_t = (1 - t) * x0 + t * noise`` and the flow-matching target is
``noise - x0``. The reference layering in Signed-RF-Playground flips both
signs; do not port coefficients across without re-deriving.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry, RegistryUnion

PolicyRole = Literal["old", "ref"]
"""Auxiliary policies an objective may require; ``current`` is always evaluated."""


@dataclass(slots=True)
class TrainPoint:
    """One re-noised endpoint. All tensors are batch-1 and float32."""

    x0: torch.Tensor
    """Clean latents (the rollout endpoint), ``[1, ...]``."""
    noise: torch.Tensor
    """Freshly drawn ``eps`` of the same shape as ``x0``."""
    t: torch.Tensor
    """``[1]`` training timestep, ``1`` = pure noise."""
    advantage: torch.Tensor
    """``[1]`` group-relative advantage of the rollout."""
    grid_index: int | None
    """Index of ``t`` on the rollout sigma grid, or ``None`` for a continuous
    draw. Decides how the trainer evaluates the point: a grid point goes through
    the rollout plan's guided step, a continuous one through the predictor."""

    @property
    def xt(self) -> torch.Tensor:
        t_expanded = self.t.view(-1, *([1] * (self.x0.ndim - 1)))
        return (1.0 - t_expanded) * self.x0 + t_expanded * self.noise

    @property
    def target(self) -> torch.Tensor:
        return self.noise - self.x0


@dataclass(slots=True)
class PolicyVelocities:
    """Velocities at the same ``(x_t, t)`` from each policy role.

    ``current`` carries gradients; ``old`` (lagged EMA policy) and ``ref``
    (frozen base) come from ``no_grad`` forwards or a cache and are detached by
    every objective, so no gradient ever reaches them.
    """

    current: torch.Tensor
    old: torch.Tensor | None = None
    ref: torch.Tensor | None = None


@dataclass(slots=True)
class LossOutput:
    loss: torch.Tensor
    """0-dim, attached to the graph."""
    metrics: dict[str, torch.Tensor]
    """0-dim detached values without prefix; the trainer adds ``train/``."""


class BaseObjective(BaseModel, ABC):
    type: str
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def rollout_policy(self) -> Literal["current", "old"]:
        """Which policy samples the rollout endpoints."""
        ...

    @abstractmethod
    def required_policies(self) -> frozenset[PolicyRole]:
        """Auxiliary velocities :meth:`compute` needs besides ``current``."""
        ...

    @abstractmethod
    def compute(self, point: TrainPoint, velocities: PolicyVelocities) -> LossOutput:
        """Per-item loss and detached metrics."""
        ...

    def _require(self, velocities: PolicyVelocities, role: PolicyRole) -> torch.Tensor:
        velocity = getattr(velocities, role)
        if velocity is None:
            raise RuntimeError(
                f"Objective {self.type!r} requires the {role!r} policy velocity "
                "but none was provided; check required_policies() against the "
                "trainer's precompute/cache path."
            )
        return velocity.detach()


objective_registry: Registry[BaseObjective] = Registry("objective", base=BaseObjective)


@objective_registry.register("nft")
class NftObjective(BaseObjective):
    """Diffusion NFT (Negative-aware FineTuning), arXiv:2509.16117.

    NFT avoids log-probability computation: the current velocity is mixed with
    the old-teacher velocity into an implicit positive policy
    ``v+ = beta * v + (1 - beta) * v_old`` and negative policy
    ``v- = (1 + beta) * v_old - beta * v``, each denoised to an ``x0`` estimate
    and regressed on the true endpoint with an adaptive per-sample L1 weight.
    The normalized advantage ``r in [0, 1]`` blends the two branches, and an
    optional MSE to the frozen reference regularises the update.
    """

    type: Literal["nft"] = "nft"
    beta: float = 1.0
    """Positive/negative prediction interpolation weight."""
    kl_beta: float = 0.01
    """KL (MSE) loss coefficient for reference model regularisation."""
    adv_clip_max: float = 5.0
    adv_mode: Literal["all", "positive_only", "negative_only", "binary"] = "all"
    """Optional advantage clipping mode."""

    def rollout_policy(self) -> Literal["current", "old"]:
        return "old"

    def required_policies(self) -> frozenset[PolicyRole]:
        roles: set[PolicyRole] = {"old"}
        if self.kl_beta > 0:
            roles.add("ref")
        return frozenset(roles)

    def compute(self, point: TrainPoint, velocities: PolicyVelocities) -> LossOutput:
        forward_prediction = velocities.current
        old_prediction = self._require(velocities, "old")
        ref_prediction = self._require(velocities, "ref") if self.kl_beta > 0 else None
        x0 = point.x0
        xt = point.xt
        t_expanded = point.t.view(-1, *([1] * (x0.ndim - 1)))

        adv = point.advantage
        adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        if self.adv_mode == "positive_only":
            adv = torch.clamp(adv, 0, self.adv_clip_max)
        elif self.adv_mode == "negative_only":
            adv = torch.clamp(adv, -self.adv_clip_max, 0)
        elif self.adv_mode == "binary":
            adv = torch.sign(adv)

        r = (adv / self.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0.0, 1.0)
        # Expand r to match spatial dims
        r = r.view(-1, *([1] * (x0.ndim - 1)))

        beta = self.beta

        # Positive & negative predictions
        positive_pred = beta * forward_prediction + (1 - beta) * old_prediction
        negative_pred = (1 + beta) * old_prediction - beta * forward_prediction

        # Predicted x0 from positive prediction
        x0_pos = xt - t_expanded * positive_pred
        with torch.no_grad():
            weight_pos = (
                torch.abs(x0_pos - x0)
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        pos_loss = ((x0_pos - x0) ** 2 / weight_pos).mean(dim=tuple(range(1, x0.ndim)))

        # Predicted x0 from negative prediction
        x0_neg = xt - t_expanded * negative_pred
        with torch.no_grad():
            weight_neg = (
                torch.abs(x0_neg - x0)
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        neg_loss = ((x0_neg - x0) ** 2 / weight_neg).mean(dim=tuple(range(1, x0.ndim)))

        # Flatten r for per-sample weighting
        r_flat = r.view(r.shape[0], -1)[:, 0]
        policy_loss_per_sample = (
            r_flat * pos_loss / beta + (1.0 - r_flat) * neg_loss / beta
        )
        policy_loss = (policy_loss_per_sample * self.adv_clip_max).mean()

        loss = policy_loss

        # KL regularisation (MSE to reference)
        if ref_prediction is not None:
            kl_loss = ((forward_prediction - ref_prediction) ** 2).mean(
                dim=tuple(range(1, x0.ndim))
            )
            kl_loss = torch.mean(kl_loss)
            loss = loss + self.kl_beta * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=forward_prediction.device)

        metrics: dict[str, torch.Tensor] = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "loss": loss.detach(),
            "old_deviate": ((forward_prediction - old_prediction) ** 2).mean().detach(),
        }
        return LossOutput(loss=loss, metrics=metrics)


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


Objective = Annotated[BaseObjective, RegistryUnion(objective_registry, "type")]


if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print

    torch.manual_seed(0)
    shape = (1, 4, 6, 6)
    point = TrainPoint(
        x0=torch.randn(shape),
        noise=torch.randn(shape),
        t=torch.tensor([0.7]),
        advantage=torch.tensor([1.5]),
        grid_index=None,
    )
    velocities = PolicyVelocities(
        current=torch.randn(shape, requires_grad=True),
        old=torch.randn(shape),
        ref=torch.randn(shape),
    )
    adapter = TypeAdapter(Objective)
    for spec in (
        "nft",
        {"type": "nft", "kl_beta": 0.0, "adv_mode": "binary"},
        "ram",
        "awm",
        {"type": "awm", "off_policy": True, "weighting": "uniform"},
        "weighted_fm",
        {"type": "weighted_fm", "eta": -1.0, "weight": "linear"},
    ):
        objective = adapter.validate_python(spec)
        out = objective.compute(point, velocities)
        (grad,) = torch.autograd.grad(out.loss, velocities.current)
        print(
            f"[bold]{spec}[/] rollout={objective.rollout_policy()} "
            f"required={sorted(objective.required_policies())} "
            f"loss={out.loss.item():.4f} |grad|={grad.norm().item():.4f}"
        )
        assert out.loss.ndim == 0 and out.loss.requires_grad
        assert all(m.ndim == 0 and not m.requires_grad for m in out.metrics.values())
        print({k: round(v.item(), 4) for k, v in out.metrics.items()})

    try:
        NftObjective().compute(point, PolicyVelocities(current=velocities.current))
    except RuntimeError as exc:
        print(f"[green]missing role raises:[/] {exc}")
    else:
        raise AssertionError("missing 'old' velocity must raise")
