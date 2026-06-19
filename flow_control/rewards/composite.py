import asyncio
import math
from typing import Any, Literal

import torch
from pydantic import ConfigDict, PrivateAttr

from flow_control.utils.logging import get_logger

from .base import BaseReward, RewardResult, reward_registry

logger = get_logger(__name__)


@reward_registry.register("composite")
class CompositeReward(BaseReward):
    """Weighted combination of multiple reward functions."""

    type: Literal["composite"] = "composite"
    rewards: list[Any]  # (weight, reward_config_dict)
    expose_internal_components: bool = True

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
            if self.expose_internal_components:
                for w in reward.component_weights:
                    weights.append(reward.weight * w)
            else:
                weights.append(reward.weight)
        return weights

    @property
    def component_labels(self) -> list[str]:
        labels: list[str] = []
        for reward in self._reward_instances:
            if not self.expose_internal_components:
                labels.append(reward.type)
                continue
            for label in reward.component_labels:
                labels.append(
                    reward.type if label == reward.type else f"{reward.type}/{label}"
                )
        return labels

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
        raise NotImplementedError(
            "CompositeReward.score() returns RewardResult directly."
        )

    def _combine_results(self, results: list[RewardResult]) -> RewardResult:
        raw_parts: list[torch.Tensor] = []
        normalized_parts: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        labels: list[str] = []
        device = results[0].normalized.device
        dtype = results[0].normalized.dtype

        for reward, result in zip(self._reward_instances, results, strict=True):
            if self.expose_internal_components:
                raw_parts.append(result.raw.to(device=device))
                normalized_parts.append(
                    result.normalized.to(device=device, dtype=dtype)
                )
                weights.append(
                    result.weights.to(device=device, dtype=dtype) * reward.weight
                )
                for label in result.labels:
                    labels.append(
                        reward.type
                        if label == reward.type
                        else f"{reward.type}/{label}"
                    )
            else:
                raw_parts.append(result.aggregate_raw().to(device=device).unsqueeze(-1))
                normalized_parts.append(
                    result.aggregate().to(device=device, dtype=dtype).unsqueeze(-1)
                )
                weights.append(
                    torch.tensor(
                        [reward.weight],
                        device=device,
                        dtype=dtype,
                    )
                )
                labels.append(reward.type)

        raw = torch.cat(raw_parts, dim=-1)
        normalized = torch.cat(normalized_parts, dim=-1)
        normalized = self.normalize.apply(normalized)
        return RewardResult(
            raw=raw,
            normalized=normalized,
            weights=torch.cat(weights, dim=0),
            labels=labels,
        )

    def score(self, batch: dict[str, Any]) -> RewardResult:
        if self.is_remote:
            result = self._remote_batch_object_call(
                "/score", batch, fields=self._batch_fields
            )
            assert isinstance(result, RewardResult)
            return result
        return self._combine_results(
            [reward.score(batch) for reward in self._reward_instances]
        )

    async def async_score(self, batch: dict[str, Any]) -> RewardResult:
        if self.is_remote:
            result = await self._async_remote_batch_object_call(
                "/score", batch, fields=self._batch_fields
            )
            assert isinstance(result, RewardResult)
            return result
        return self._combine_results(await self._async_score_children(batch))

    async def _async_score_children(self, batch: dict[str, Any]) -> list[RewardResult]:
        """Score children concurrently before ``_combine_results`` concatenates."""
        scores = await asyncio.gather(
            *(r.async_score(batch) for r in self._reward_instances)
        )
        return list(scores)

    def supports_rollout_overlap(self) -> bool:
        return all(
            reward.supports_rollout_overlap() for reward in self._reward_instances
        )

    def _unload_model(self) -> None:
        for reward in self._reward_instances:
            reward.unload_model()
