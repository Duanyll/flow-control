from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, PlainValidator


class AdvantageEstimator(BaseModel, ABC):
    """Advantage normalization strategy."""

    type: str
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def compute(
        self,
        rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        num_timesteps: int,
    ) -> torch.Tensor:
        """Compute advantage weights.

        Args:
            rewards: [B] per-sample rewards (gathered across all GPUs).
            prompt_ids: [B] integer IDs identifying which prompt each sample belongs to.
            num_timesteps: T — the number of timesteps to expand the result to.

        Returns:
            Tensor of shape [B, T] with per-sample, per-timestep advantages.
        """
        ...


class PerPromptAdvantage(AdvantageEstimator):
    """Per-prompt normalization: normalize within each prompt group."""

    # Narrow the discriminator field for Pydantic; safe because Pydantic handles this at runtime.
    type: Literal["per_prompt"] = "per_prompt"  # type: ignore[assignment]
    use_global_std: bool = True
    eps: float = 1e-4

    def compute(
        self,
        rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        num_timesteps: int,
    ) -> torch.Tensor:
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

        return advantages.unsqueeze(1).expand(-1, num_timesteps)


class GlobalAdvantage(AdvantageEstimator):
    """Global normalization: (r - mean) / (std + eps)."""

    # Narrow the discriminator field for Pydantic; safe because Pydantic handles this at runtime.
    type: Literal["global"] = "global"  # type: ignore[assignment]
    eps: float = 1e-4

    def compute(
        self,
        rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        num_timesteps: int,
    ) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std(correction=0) + self.eps
        advantages = (rewards - mean) / std
        return advantages.unsqueeze(1).expand(-1, num_timesteps)


ADVANTAGE_REGISTRY: dict[str, type[AdvantageEstimator]] = {
    "per_prompt": PerPromptAdvantage,
    "global": GlobalAdvantage,
}


def parse_advantage(conf: dict | AdvantageEstimator) -> AdvantageEstimator:
    """Parse an advantage estimator from a dict or pass through an existing instance."""
    if isinstance(conf, AdvantageEstimator):
        return conf
    adv_type = conf["type"]
    adv_class = ADVANTAGE_REGISTRY.get(adv_type)
    if adv_class is None:
        raise ValueError(f"Unknown advantage type: {adv_type}")
    return adv_class(**conf)


Advantage = Annotated[AdvantageEstimator, PlainValidator(parse_advantage)]
