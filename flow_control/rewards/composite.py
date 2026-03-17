import asyncio
import math
from typing import Any, Literal

import torch
from pydantic import ConfigDict, PrivateAttr

from flow_control.utils.logging import get_logger

from .base import BaseReward

logger = get_logger(__name__)


class CompositeReward(BaseReward):
    """Weighted combination of multiple reward functions."""

    type: Literal["composite"] = "composite"
    rewards: list[Any]  # (weight, reward_config_dict)

    model_config = ConfigDict(extra="forbid")

    _reward_instances: list[BaseReward] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        from . import parse_reward

        if not self.rewards:
            raise ValueError("CompositeReward requires at least one child reward.")

        self._reward_instances = []
        total_weight = 0.0
        for reward_conf in self.rewards:
            if isinstance(reward_conf, dict):
                self._reward_instances.append(parse_reward(reward_conf))
            elif isinstance(reward_conf, BaseReward):
                self._reward_instances.append(reward_conf)
            total_weight += self._reward_instances[-1].weight
        if not math.isclose(total_weight, 1.0):
            logger.warning(
                f"CompositeReward total weight is {total_weight:.3f}, not 1.0. "
                "Consider normalizing weights for interpretability."
            )

    @property
    def component_weights(self) -> list[float]:
        """Flatten children's component weights, scaled by each child's weight."""
        weights: list[float] = []
        for reward in self._reward_instances:
            for w in reward.component_weights:
                weights.append(reward.weight * w)
        return weights

    @property
    def _batch_fields(self) -> set[str]:
        fields = set()
        for reward in self._reward_instances:
            fields.update(reward._batch_fields)
        return fields

    def _load_model(self, device: torch.device) -> None:
        for reward in self._reward_instances:
            reward.load_model(device)

    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Concatenate children's per-component scores into ``[total_C]``."""
        parts: list[torch.Tensor] = []
        for reward in self._reward_instances:
            parts.append(reward.score(batch))
        return torch.cat(parts, dim=0)

    async def async_score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Async version that scores children concurrently, returns ``[total_C]``."""
        if self.is_remote:
            return await self._async_remote_batch_call(
                "/score", batch, fields=self._batch_fields
            )
        scores = await asyncio.gather(
            *(r.async_score(batch) for r in self._reward_instances)
        )
        return torch.cat(list(scores), dim=0)

    def supports_rollout_overlap(self) -> bool:
        return all(
            reward.supports_rollout_overlap() for reward in self._reward_instances
        )

    def _unload_model(self) -> None:
        for reward in self._reward_instances:
            reward.unload_model()
