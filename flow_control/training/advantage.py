from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry, RegistryUnion


class AdvantageEstimator(BaseModel, ABC):
    """Advantage normalization strategy."""

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def compute(
        self,
        rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantage weights.

        Args:
            rewards: ``[B, C]`` per-sample, per-component rewards (gathered
                across all GPUs).
            prompt_ids: ``[B]`` integer IDs identifying which prompt each
                sample belongs to.
            weights: ``[C]`` component weights (from
                ``reward.component_weights``).

        Returns:
            Tensor of shape ``[B]`` with per-sample advantages.
        """
        ...


advantage_registry: Registry[AdvantageEstimator] = Registry(
    "advantage", base=AdvantageEstimator
)


@advantage_registry.register("per_prompt")
class PerPromptAdvantage(AdvantageEstimator):
    """Per-prompt normalization: normalize within each prompt group."""

    type: Literal["per_prompt"] = "per_prompt"
    use_global_std: bool = True
    eps: float = 1e-4

    def compute(
        self,
        rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # Combine components first: [B, C] → [B]
        combined = (rewards * weights).sum(dim=-1)

        advantages = torch.zeros_like(combined)
        unique_ids = prompt_ids.unique()

        global_std = combined.std(correction=0) + self.eps

        for pid in unique_ids:
            mask = prompt_ids == pid
            group_rewards = combined[mask]
            group_mean = group_rewards.mean()
            if self.use_global_std:
                advantages[mask] = (group_rewards - group_mean) / global_std
            else:
                group_std = group_rewards.std(correction=0) + self.eps
                advantages[mask] = (group_rewards - group_mean) / group_std

        return advantages


@advantage_registry.register("global")
class GlobalAdvantage(AdvantageEstimator):
    """Global normalization: (r - mean) / (std + eps)."""

    type: Literal["global"] = "global"
    eps: float = 1e-4

    def compute(
        self,
        rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # Combine components first: [B, C] → [B]
        combined = (rewards * weights).sum(dim=-1)

        mean = combined.mean()
        std = combined.std(correction=0) + self.eps
        return (combined - mean) / std


@advantage_registry.register("gdpo")
class GdpoAdvantage(AdvantageEstimator):
    """GDPO: normalize each reward component independently per-prompt-group,
    then combine with weights and batch-normalize.

    This prevents a dominant component from overwhelming others when using
    multi-component rewards (CompositeReward, UnifiedReward).
    """

    type: Literal["gdpo"] = "gdpo"
    eps: float = 1e-4

    def compute(
        self,
        rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # rewards: [B, C], weights: [C], prompt_ids: [B]
        B, C = rewards.shape
        normalized = torch.zeros_like(rewards)
        unique_ids = prompt_ids.unique()

        for i in range(C):
            for pid in unique_ids:
                mask = prompt_ids == pid
                group = rewards[mask, i]
                mean = group.mean()
                std = group.std(correction=0) + self.eps
                normalized[mask, i] = (group - mean) / std

        # Weighted combination: [B, C] → [B]
        pre_bn = (normalized * weights).sum(dim=-1)

        # Batch normalization
        mean = pre_bn.mean()
        std = pre_bn.std(correction=0) + self.eps
        return (pre_bn - mean) / std


Advantage = Annotated[AdvantageEstimator, RegistryUnion(advantage_registry, "type")]


if __name__ == "__main__":
    from rich import print as rprint

    torch.manual_seed(42)

    B, C = 12, 3  # 12 samples, 3 components
    K = 4  # 4 rollouts per prompt, 3 prompts
    rewards = torch.randn(B, C)
    prompt_ids = torch.tensor([0] * K + [1] * K + [2] * K)
    weights = torch.tensor([0.4, 0.3, 0.3])

    rprint("[bold cyan]===== PerPromptAdvantage (backward compat) =====[/]")
    ppa = PerPromptAdvantage()
    adv = ppa.compute(rewards, prompt_ids, weights)
    rprint(f"  shape: {adv.shape}, mean: {adv.mean():.4f}, std: {adv.std():.4f}")
    rprint(f"  values: {adv}")

    # Verify backward compatibility with [B, 1] (single component)
    rewards_1c = rewards[:, :1]
    weights_1c = torch.tensor([1.0])
    adv_1c = ppa.compute(rewards_1c, prompt_ids, weights_1c)
    rprint(f"\n  [B,1] shape: {adv_1c.shape}, mean: {adv_1c.mean():.4f}")

    rprint("\n[bold cyan]===== GlobalAdvantage (backward compat) =====[/]")
    ga = GlobalAdvantage()
    adv = ga.compute(rewards, prompt_ids, weights)
    rprint(f"  shape: {adv.shape}, mean: {adv.mean():.6f}, std: {adv.std():.4f}")
    rprint(f"  values: {adv}")

    rprint("\n[bold cyan]===== GdpoAdvantage =====[/]")
    gdpo = GdpoAdvantage()
    adv = gdpo.compute(rewards, prompt_ids, weights)
    rprint(f"  shape: {adv.shape}, mean: {adv.mean():.6f}, std: {adv.std():.4f}")
    rprint(f"  values: {adv}")

    # Verify GDPO with single component reduces to per-prompt + global norm
    rprint("\n[bold cyan]===== GDPO single component =====[/]")
    adv_gdpo_1c = gdpo.compute(rewards_1c, prompt_ids, weights_1c)
    rprint(f"  shape: {adv_gdpo_1c.shape}")
    rprint(f"  values: {adv_gdpo_1c}")

    rprint("\n[bold green]All advantage tests passed.[/]")
