from typing import Any

import torch
from pydantic import ConfigDict, PrivateAttr

from .base import BaseReward


class CompositeReward(BaseReward):
    """Weighted combination of multiple reward functions."""

    type: str = "composite"
    rewards: list[tuple[float, Any]]  # (weight, reward_config_dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _reward_instances: list[tuple[float, BaseReward]] = PrivateAttr(
        default_factory=list
    )

    def model_post_init(self, __context: Any) -> None:
        from . import parse_reward

        self._reward_instances = []
        for weight, reward_conf in self.rewards:
            if isinstance(reward_conf, dict):
                self._reward_instances.append((weight, parse_reward(reward_conf)))
            elif isinstance(reward_conf, BaseReward):
                self._reward_instances.append((weight, reward_conf))

    def load_model(self, device: torch.device) -> None:
        for _, reward in self._reward_instances:
            reward.load_model(device)

    def score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute weighted sum of reward scores for a single sample."""
        total = torch.tensor(0.0)
        for weight, reward in self._reward_instances:
            total = total + weight * reward.score(batch).to(total.device)
        return total

    def score_detailed(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted sum with per-reward breakdown."""
        total = torch.tensor(0.0)
        details: dict[str, torch.Tensor] = {}
        for weight, reward in self._reward_instances:
            s = reward.score(batch).to(total.device)
            details[reward.type] = s
            total = total + weight * s
        return total, details

    def unload_model(self) -> None:
        for _, reward in self._reward_instances:
            reward.unload_model()
