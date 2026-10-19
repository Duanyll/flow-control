"""Diffusion NFT (Negative-aware FineTuning), arXiv:2509.16117.

:class:`NftObjective` is the math, :class:`NftTrainer` the
:class:`~flow_control.training.endpoint.EndpointTrainer` preset: rollouts are
sampled by the ``old`` variant (an EMA stepped once per epoch).
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
    sigma_like,
)
from .mixins import trainer_registry
from .train_timesteps import GridTimesteps, TrainTimesteps


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

    def rollout_variant(self) -> str:
        return "old"

    def required_variants(self) -> frozenset[str]:
        return frozenset({"old", "ref"} if self.kl_beta > 0 else {"old"})

    def compute(
        self, point: Row, current: torch.Tensor, aux: dict[str, torch.Tensor]
    ) -> LossOutput:
        forward_prediction = current
        old_prediction = aux["old"].detach()
        ref_prediction = aux["ref"].detach() if self.kl_beta > 0 else None
        x0 = point["clean_latents"]
        xt = point["noisy_latents"]
        t_expanded = sigma_like(point, x0)

        adv = point["advantage"]
        adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        if self.adv_mode == "positive_only":
            adv = torch.clamp(adv, 0, self.adv_clip_max)
        elif self.adv_mode == "negative_only":
            adv = torch.clamp(adv, -self.adv_clip_max, 0)
        elif self.adv_mode == "binary":
            adv = torch.sign(adv)

        r = (adv / self.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0.0, 1.0)
        r = r.view(-1, *([1] * (x0.ndim - 1)))

        beta = self.beta
        positive_pred = beta * forward_prediction + (1 - beta) * old_prediction
        negative_pred = (1 + beta) * old_prediction - beta * forward_prediction

        x0_pos = xt - t_expanded * positive_pred
        with torch.no_grad():
            weight_pos = (
                torch.abs(x0_pos - x0)
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        pos_loss = ((x0_pos - x0) ** 2 / weight_pos).mean(dim=tuple(range(1, x0.ndim)))

        x0_neg = xt - t_expanded * negative_pred
        with torch.no_grad():
            weight_neg = (
                torch.abs(x0_neg - x0)
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        neg_loss = ((x0_neg - x0) ** 2 / weight_neg).mean(dim=tuple(range(1, x0.ndim)))

        r_flat = r.view(r.shape[0], -1)[:, 0]
        policy_loss_per_sample = (
            r_flat * pos_loss / beta + (1.0 - r_flat) * neg_loss / beta
        )
        policy_loss = (policy_loss_per_sample * self.adv_clip_max).mean()

        loss = policy_loss
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


@trainer_registry.register("nft")
class NftTrainer(EndpointTrainer):
    training_type: str = "nft"
    objective: Objective = NftObjective()
    train_timesteps: TrainTimesteps = GridTimesteps()


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
    for objective in (NftObjective(), NftObjective(kl_beta=0.0, adv_mode="binary")):
        out = objective.compute(point, current, aux)
        (grad,) = torch.autograd.grad(out.loss, current)
        assert out.loss.ndim == 0 and out.loss.requires_grad
        assert all(m.ndim == 0 and not m.requires_grad for m in out.metrics.values())
        print(objective, f"required={sorted(objective.required_variants())}")
        print(
            f"|grad|={grad.norm().item():.4f}",
            {k: round(v.item(), 4) for k, v in out.metrics.items()},
        )
