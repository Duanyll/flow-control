import re
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.processors.components.llm import LLMClient
from flow_control.utils.logging import get_logger

from .base import BaseReward

logger = get_logger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "You are presented with a generated image and its associated text caption. "
    "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
    "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
    "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Coherence Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [{prompt}]"
)


class ScoreTag(BaseModel):
    """A named score axis to extract from the LLM output."""

    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = 1.0


class UnifiedReward(BaseReward):
    """LLM-based reward that uses a vision-language model to judge images.

    This reward is **async-only**: the synchronous ``_score`` raises
    ``NotImplementedError``.  Use ``async_score`` (or the async GRPO
    rollout path) instead.
    """

    type: Literal["unified_reward"] = "unified_reward"
    llm: LLMClient

    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    score_tags: list[ScoreTag] = [
        ScoreTag(name="Alignment Score"),
        ScoreTag(name="Coherence Score"),
        ScoreTag(name="Style Score"),
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _load_model(self, device: torch.device) -> None:
        pass

    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError(
            "UnifiedReward is async-only. Use async_score() instead."
        )

    def _parse_scores(self, output: str) -> dict[str, float]:
        """Extract score values from LLM output text."""
        scores: dict[str, float] = {}
        for tag in self.score_tags:
            pattern = rf"{re.escape(tag.name)}\s*\(1-5\)\s*:\s*([\d.]+)"
            match = re.search(pattern, output)
            if match:
                scores[tag.name] = float(match.group(1))
            else:
                logger.warning(
                    "Failed to parse score for tag '%s' from LLM output: %s",
                    tag.name,
                    output[:200],
                )
                scores[tag.name] = 0.0
        return scores

    async def async_score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Score an image using the LLM judge."""
        prompt_text: str = batch["prompt"]
        image: torch.Tensor = batch["clean_image"]

        user_prompt = self.prompt_template.format(prompt=prompt_text)
        llm_output, _ = await self.llm.generate(user_prompt, images=[image])

        scores = self._parse_scores(llm_output)

        total_weight = sum(tag.weight for tag in self.score_tags)
        weighted_sum = sum(scores[tag.name] * tag.weight for tag in self.score_tags)
        # Normalize to ~[0, 1] range (scores are 1-5, divide by 5)
        reward = weighted_sum / (total_weight * 5.0) if total_weight > 0 else 0.0

        return torch.tensor(reward, dtype=torch.float32)

    def supports_rollout_overlap(self) -> bool:
        return True
