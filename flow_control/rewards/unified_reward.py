import re
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from flow_control.processors.components.llm import LLMClient
from flow_control.utils.logging import get_logger

from .base import BaseReward
from .normalize import AffineNormalize, Normalize

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
    # Raw scores are in [1, 5]; default normalize divides by 5 to land in
    # [0.2, 1.0], matching the previous hard-coded behaviour.
    normalize: Normalize = Field(default_factory=lambda: AffineNormalize(scale=0.2))

    model_config = ConfigDict(extra="forbid")

    @property
    def component_weights(self) -> list[float]:
        return [tag.weight for tag in self.score_tags]

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

    async def _async_score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Score an image using the LLM judge.

        Returns:
            Tensor of shape ``[C]`` with raw per-tag scores in ``[1, 5]``.
            The base class applies :attr:`normalize` on top.  Configure
            ``normalize`` with e.g. ``{"type": "affine", "scale": 0.2}`` to
            land back in ``[0.2, 1.0]`` as before.
        """
        prompt_text: str = batch["prompt"]
        image: torch.Tensor = batch["clean_image"]

        user_prompt = self.prompt_template.format(prompt=prompt_text)
        llm_output, _ = await self.llm.generate(user_prompt, images=[image])

        scores = self._parse_scores(llm_output)

        per_tag = [scores[tag.name] for tag in self.score_tags]
        return torch.tensor(per_tag, dtype=torch.float32)

    def supports_rollout_overlap(self) -> bool:
        return True
