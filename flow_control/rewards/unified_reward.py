import re
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from flow_control.processors.components.llm import LLMClient
from flow_control.samplers.node import Row
from flow_control.utils.logging import get_logger

from .base import (
    BaseJudge,
    BaseScorer,
    RemoteReward,
    judge_registry,
    scorer_registry,
)
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

# The official pairwise templates, verbatim; the images go before the text.
PAIR_SCORE_TEMPLATE = (  # UnifiedReward-2.0 pair_score_image_generation.py
    "You are presented with two generated images (Image 1 and Image 2) along with a shared text caption. "
    "Your task is to comparatively evaluate the two images across three specific dimensions:\n\n"
    "- Alignment Score: How well each image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent and visually coherent each image is (absence of visual glitches, distorted objects, etc.).\n"
    "- Style Score: How aesthetically appealing each image is, regardless of caption accuracy.\n\n"
    "For each dimension, you must assign a relative score to Image 1 and Image 2, such that:\n"
    "- Each score is a float between 0 and 1 (inclusive).\n"
    "- The scores for Image 1 and Image 2 must sum to exactly 1.0 for each dimension.\n"
    "- The higher the score, the better that image is in the corresponding dimension *relative to the other*.\n\n"
    "This format emphasizes comparative quality rather than absolute evaluation.\n\n"
    "Please provide your evaluation in the format below:\n\n"
    "Alignment Score:\n"
    " Image 1: X\n"
    " Image 2: Y\n\n"
    "Coherence Score:\n"
    " Image 1: X\n"
    " Image 2: Y\n\n"
    "Style Score:\n"
    " Image 1: X\n"
    " Image 2: Y\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [{prompt}]"
)
PAIR_RANK_TEMPLATE = (  # qwen_pair_rank_image_generation.py
    "You are given a text caption and two generated images based on that caption. "
    "Your task is to evaluate and compare these images based on two key criteria:\n"
    "1. Alignment with the Caption: Assess how well each image aligns with the provided caption. "
    "Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
    "2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, "
    "color accuracy, and overall aesthetic appeal.\n"
    "Compare both images using the above criteria and select the one that better aligns with the caption "
    "while exhibiting superior visual quality.\n"
    'Provide a clear conclusion such as "Image 1 is better.", "Image 2 is better." '
    'and "Both images are equally good."\n'
    "Your task is provided as follows:\n"
    "Text Caption: [{prompt}]"
)
PAIR_DIMENSIONS = ("Alignment Score", "Coherence Score", "Style Score")
PAIR_VERDICTS = {
    "image 1 is better": 1.0,
    "image 2 is better": 0.0,
    "both images are equally good": 0.5,
}
_NUMBER = r"\d*\.?\d+"


class ScoreTag(BaseModel):
    """A named score axis to extract from the LLM output."""

    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = 1.0


@scorer_registry.register("unified_reward")
class UnifiedRewardScorer(BaseScorer, RemoteReward[Row, Tensor]):
    """LLM-based reward that uses a vision-language model to judge images,
    one request per row; one component per score tag."""

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
    def labels(self) -> list[str]:
        return [tag.name.lower().replace(" ", "_") for tag in self.score_tags]

    @property
    def weights(self) -> list[float]:
        return [tag.weight for tag in self.score_tags]

    @property
    def fields(self) -> set[str]:
        return {"clean_image", "prompt"}

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

    async def run_one(self, row: Row) -> Tensor:
        """Score an image using the LLM judge.

        Returns:
            Tensor of shape ``[C]`` with raw per-tag scores in ``[1, 5]``;
            :attr:`normalize` (default affine 0.2) maps them to ``[0.2, 1.0]``.
        """
        prompt_text: str = row["prompt"]
        image: torch.Tensor = row["clean_image"]

        user_prompt = self.prompt_template.format(prompt=prompt_text)
        llm_output, _ = await self.llm.generate(user_prompt, images=[image])

        scores = self._parse_scores(llm_output)

        per_tag = [scores[tag.name] for tag in self.score_tags]
        return torch.tensor(per_tag, dtype=torch.float32)


@judge_registry.register("unified_reward")
class UnifiedRewardJudge(BaseJudge, RemoteReward[tuple[Row, Row], float]):
    """MLLM pair judge on UnifiedReward's official pairwise templates, one
    request per pair: Image 1 is ``a``, Image 2 is ``b``, both before the text
    as in the official scripts.

    ``mode`` picks the template the checkpoint was trained on:

    - ``"score"``: the UnifiedReward-2.0 "pair score" template
      (``UnifiedReward-2.0-qwen-*``): relative Alignment / Coherence / Style
      scores of the two images, summing to 1; P(a ≻ b) is the mean over the
      three of ``X / (X + Y)``.
    - ``"rank"``: the classic "pair rank" template (``UnifiedReward-qwen-7b`` /
      ``UnifiedReward-7b-v1.5``): the last of "Image 1 is better." / "Image 2
      is better." / "Both images are equally good." gives 1 / 0 / 0.5.

    An answer that cannot be parsed scores 0.5, with a warning. The judge
    answers one order; ``round_robin`` asks both itself (``swap_orders``).
    """

    type: Literal["unified_reward"] = "unified_reward"
    llm: LLMClient
    mode: Literal["score", "rank"] = "score"

    @property
    def fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _parse(self, output: str) -> float | None:
        """P(Image 1 ≻ Image 2) read from *output*; ``None`` if it cannot be parsed."""
        if self.mode == "rank":
            verdicts = re.findall("|".join(PAIR_VERDICTS), output, re.IGNORECASE)
            return PAIR_VERDICTS[verdicts[-1].lower()] if verdicts else None
        shares: list[float] = []
        for dimension in PAIR_DIMENSIONS:
            match = re.search(
                rf"{dimension}\W*Image 1\W*({_NUMBER})\W*Image 2\W*({_NUMBER})",
                output,
                re.IGNORECASE,
            )
            if match is None:
                return None
            x, y = float(match[1]), float(match[2])
            shares.append(x / (x + y) if x + y > 0 else 0.5)
        return sum(shares) / len(shares)

    async def run_one(self, pair: tuple[Row, Row]) -> float:
        a, b = pair
        template = PAIR_SCORE_TEMPLATE if self.mode == "score" else PAIR_RANK_TEMPLATE
        output, _ = await self.llm.generate(
            template.format(prompt=a["prompt"]),
            images=[a["clean_image"], b["clean_image"]],
        )
        preference = self._parse(output)
        if preference is None:
            logger.warning(
                "Failed to parse a UnifiedReward pair %s verdict; scoring the pair "
                "0.5. LLM output: %s",
                self.mode,
                output[:200],
            )
            return 0.5
        return preference


if __name__ == "__main__":
    import asyncio
    from collections.abc import Callable
    from typing import TYPE_CHECKING, cast

    from rich import print

    from flow_control.processors.components.llm import ImageContent, TextContent
    from flow_control.training.rollouts.pairwise import preferences

    from .base import Reward

    if TYPE_CHECKING:
        from flow_control.training.rollouts.runner import RewardContext

    calls: list[tuple[str, list[Tensor]]] = []

    class FakeLLM(LLMClient):
        """Answers ``reply(images)``; records every ``(prompt, images)``."""

        reply: Callable[[list[Tensor]], str]

        async def generate(
            self,
            user_prompt: str | list[TextContent | ImageContent],
            images: list[Tensor] | None = None,
            system_prompt: str = "You are a helpful assistant.",
            context: list | None = None,
            strip_think: bool = True,
        ) -> tuple[str, list]:
            assert isinstance(user_prompt, str) and images is not None
            calls.append((user_prompt, images))
            return self.reply(images), []

    def judge(mode: Literal["score", "rank"], reply: Callable[[list[Tensor]], str]):
        return UnifiedRewardJudge(llm=FakeLLM(reply=reply), mode=mode)

    def pair_score_output(share_1: list[float]) -> str:
        return "\n".join(
            f"{d}:\n Image 1: {s}\n Image 2: {1 - s}\n"
            for d, s in zip(PAIR_DIMENSIONS, share_1, strict=True)
        )

    # The example output in the official pair_score_image_generation.py.
    official = """
Alignment Score:
 Image 1: 0.435699999332428
 Image 2: 0.564300000667572

Coherence Score:
 Image 1: 0.4180000126361847
 Image 2: 0.5820000171661377

Style Score:
 Image 1: 0.40790000557899475
 Image 2: 0.5921000242233276
"""
    caption = "a red cube on a wooden table"
    a: Row = {"prompt": caption, "clean_image": torch.full((3, 8, 8), 0.2)}
    b: Row = {"prompt": caption, "clean_image": torch.full((3, 8, 8), 0.6)}

    # score: the mean over the three dimensions of X / (X + Y), Image 1 = a
    # (Style sums to 1.00000003: the raw mean of X would be 0.42053333918).
    p = asyncio.run(judge("score", lambda _: official).run_one((a, b)))
    assert abs(p - 0.4205333309779566) < 1e-12, p
    markdown = official.replace("Style Score:", "**style score:**")  # same parse
    assert asyncio.run(judge("score", lambda _: markdown).run_one((a, b))) == p
    prompt, images = calls[-1]
    assert prompt == PAIR_SCORE_TEMPLATE.format(prompt=caption), prompt
    assert prompt.endswith(f"Text Caption: [{caption}]"), prompt
    assert images[0] is a["clean_image"] and images[1] is b["clean_image"]

    # rank: the last conclusion sentence decides, case-insensitively.
    for output, expected in [
        ("Image 1 is better.", 1.0),
        (
            "Image 1 is better aligned, but its cube is distorted. Overall, "
            "image 2 is better.",
            0.0,
        ),
        ("Both images are equally good.", 0.5),
    ]:
        p = asyncio.run(judge("rank", lambda _, o=output: o).run_one((a, b)))
        assert p == expected, (output, p)
    assert calls[-1][0] == PAIR_RANK_TEMPLATE.format(prompt=caption), calls[-1][0]

    # Unparseable: garbage, or a score answer missing a dimension, is a tie.
    unparseable: list[tuple[Literal["score", "rank"], str]] = [
        ("rank", "I cannot tell."),
        ("score", "I cannot tell."),
        ("score", official.split("Coherence")[0]),
    ]
    for mode, output in unparseable:
        p = asyncio.run(judge(mode, lambda _, o=output: o).run_one((a, b)))
        assert p == 0.5, (mode, output, p)

    # Through preferences(swap_orders=True): a judge favouring Image 1 by 0.1
    # on every dimension folds back to q_a / (q_a + q_b).
    def biased(images: list[Tensor]) -> str:
        q_1, q_2 = (float(image.mean()) for image in images)
        return pair_score_output([q_1 / (q_1 + q_2) + 0.1] * 3)

    class _Ctx:
        """Stands in for the runner's RewardContext."""

        async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]:
            assert isinstance(reward, RemoteReward), reward
            return list(await asyncio.gather(*(reward.run_one(i) for i in items)))

    stub = cast("RewardContext", _Ctx())
    calls.clear()
    folded = asyncio.run(preferences(stub, judge("score", biased), [(a, b)], True))
    assert torch.allclose(folded, torch.tensor([0.25])), folded
    assert [[i is a["clean_image"] for i in images] for _, images in calls] == [
        [True, False],
        [False, True],
    ], "asks (a, b) then (b, a)"

    print("[green]rewards.unified_reward self-test passed[/green]")
