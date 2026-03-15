from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Tag


class AdvantageEstimator(BaseModel, ABC):
    """Advantage normalization strategy."""

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def compute(self, rewards: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Compute advantage weights.

        Args:
            rewards: [B] per-sample rewards (gathered across all GPUs).
            prompt_ids: [B] integer IDs identifying which prompt each sample belongs to.

        Returns:
            Tensor of shape [B] with per-sample, per-timestep advantages.
        """
        ...


class PerPromptAdvantage(AdvantageEstimator):
    """Per-prompt normalization: normalize within each prompt group."""

    type: Literal["per_prompt"] = "per_prompt"
    use_global_std: bool = True
    eps: float = 1e-4

    def compute(self, rewards: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        unique_ids = prompt_ids.unique()

        global_std = rewards.std(correction=0) + self.eps

        for pid in unique_ids:
            mask = prompt_ids == pid
            group_rewards = rewards[mask]
            group_mean = group_rewards.mean()
            if self.use_global_std:
                advantages[mask] = (group_rewards - group_mean) / global_std
            else:
                group_std = group_rewards.std(correction=0) + self.eps
                advantages[mask] = (group_rewards - group_mean) / group_std

        return advantages


class GlobalAdvantage(AdvantageEstimator):
    """Global normalization: (r - mean) / (std + eps)."""

    type: Literal["global"] = "global"
    eps: float = 1e-4

    def compute(self, rewards: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std(correction=0) + self.eps
        advantages = (rewards - mean) / std
        return advantages


def _passthrough_advantage(v: object) -> object:
    """Allow already-instantiated AdvantageEstimator instances to pass through."""
    if isinstance(v, AdvantageEstimator):
        return v
    return v


_AdvantageUnion = Annotated[
    Annotated[PerPromptAdvantage, Tag("per_prompt")]
    | Annotated[GlobalAdvantage, Tag("global")],
    Discriminator("type"),
]

Advantage = Annotated[_AdvantageUnion, BeforeValidator(_passthrough_advantage)]
