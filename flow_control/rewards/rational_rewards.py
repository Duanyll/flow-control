"""RationalRewards T2I reward — an API-based VLM-judge reward.

This ports the ``rational_rewards_t2i`` reward from Flow-Factory.  It calls an
OpenAI-compatible vLLM server hosting the ``RationalRewards-8B-T2I`` judge model,
which scores a generated image on three rubric aspects (1-4 scale) and emits a
structured text response.  The aspect scores are parsed out, averaged
(``N/A``-aware), and mapped to ``[0, 1]``.

The aggregated score is the primary reward component (drives the advantage); the
individual aspect sub-scores are exposed as additional *zero-weight* components
so they show up in ``rollout/raw/<aspect>_*`` / ``rollout/normalized/<aspect>_*``
metrics without affecting optimization.
"""

import asyncio
import re
from typing import Any, Literal

import torch
from pydantic import ConfigDict

from flow_control.processors.components.llm import (
    ImageContent,
    LLMClient,
    TextContent,
)
from flow_control.utils.logging import get_logger

from .base import BaseReward

logger = get_logger(__name__)

# Aspect key -> (numbered-block prefix, header substring) used during parsing.
SUPPORTED_ASPECTS: dict[str, tuple[str, str]] = {
    "text_faithfulness": ("1.", "Text Faithfulness"),
    "physical_quality": ("2.", "Physical and Visual Quality"),
    "text_rendering": ("3.", "Text Rendering"),
}

RATIONAL_T2I_SYSTEM_PROMPT = (
    "You are an expert image generation evaluator. Your task is to evaluate the "
    "quality of a generated image based on a user instruction. Afterwards, you "
    "need to suggest how to refine the original user request to produce better "
    "image generation (if any)."
)

# Verbatim rubric suffix from Flow-Factory (kept as one literal, no ``<image>``
# token so user prompts may freely contain that text).
T2I_SCORING_PROMPT_SUFFIX = """

To do this, you must first assess the image on three critical aspects, provide justifications and absolute scores in 1-4 scale.

### Critical Aspects & Scoring Rubric
**1. Text Faithfulness** (How accurately does the output follow the instruction?)
- **4 (Full match):** All key elements (objects, colors, actions) are represented exactly as described. No hallucinations or unrequested changes.
- **3 (Minor mismatch):** Most key elements are present, but minor details are missing, incorrect, or slightly inaccurate.
- **2 (Some mismatch):** Some key elements are missing, altered, or interpreted incorrectly.
- **1 (Major deviations):** Key elements are completely missing, altered, or contradicted. Instruction is ignored.

**2. Physical and Visual Quality** (Technical errors, composition, realism, and physics)
- **4 (No noticeable flaws):** The image is physically plausible (correct lighting, shadows, geometry, anatomy). No visible artifacts (seams, blurring, noise).
- **3 (Minor flaws):** Small inaccuracies that are noticeable but not strongly disruptive (e.g., slight lighting mismatch, minor texture issues).
- **2 (Some flaws):** Clear physical or visual errors that disrupt the image (e.g., incorrect perspective, "floating" objects, wrong shadow direction, obvious seams).
- **1 (Severe flaws):** Major physical/visual errors (e.g., impossible geometry, distorted anatomy, garbled objects, severe artifacts).

**3. Text Rendering** (Only if the instruction involves generating text)
- **4 (Full match):** Text is correct, legible, and integrated well.
- **3 (Mostly match):** Minor misspellings or inconsistent capitalization.
- **2 (Partial match):** Major misspellings or distorted text.
- **1 (Major deviations):** Text is unreadable, severely distorted, or missing. (Use N/A if no text generation is required).

### Scoring Methodology (CRITICAL)
During assessment for each aspect, recall the initial user request and the scoring rubrics of the aspect, provide scores with detailed justifications for the generated image and reflect fine-grained preferences.
1. **Anchor:** Have a global inspection based on the user request and the resulting generation. Determine the rough integer score level (1, 2, 3, or 4) according to the definitions provided.
2. **Justify and Adjust:** Do careful visual analysis and identify specific flaws in generation. Justify the score with concrete evidence and scoring logic. Fine-tune this anchor score into a float value. Add small increments for exceptional execution or deduct points for specific flaws.
   - *Example:* deduct points from 4.0 for slight flaws if the assessed dimension is close to satisfaction. add increments from 1.0 or 2.0 based on severity of flaws.

Afterwards, try to construct a refined user request that helps the visual generation model to produce better image generation.
Think of the weaknesses identified in the judgement, then map them to instruction details and apply specific fixes.
Provide a final new user request that enrich the initial user request.

Output your evaluation in the following format:
# User Request Analysis
[ understanding the user request, try to analyze or decompose the user request deeper. Think of what the request might imply or what needs to be inferred to successfully execute the request. ]
# Detailed Judgement
1. Text Faithfulness:
## Justification: [ Analysis of the user request and the assessment of the resulting generation. How it comes to a final score. ]
## Score: [ float score ]
2. Physical and Visual Quality:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score ]
3. Text Rendering:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score or N/A ]
# Summary: [ Summary of the evaluation ]

# User Request Refinement:
## Refinement Comments: [Specific suggestions for improving the user request]
## Refined Request: [The improved, more specific user request for generation like a standard user instruction]"""

_SCORE_LINE_RE = re.compile(r"(?:##\s*)?Score\s*:\s*(.+)$", re.IGNORECASE)
_LEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def _extract_numeric_score(value: str) -> float | str:
    """Parse a captured score string into a float, or the ``"N/A"`` sentinel."""
    stripped = value.strip()
    if stripped.upper().startswith("N/A"):
        return "N/A"
    match = _LEADING_NUMBER_RE.match(stripped)
    if match is None:
        raise ValueError(f"Could not extract a numeric score from {value!r}")
    return float(match.group(1))


def _extract_score_from_block(block: list[str]) -> float | str | None:
    for line in block:
        match = _SCORE_LINE_RE.search(line.strip())
        if match is not None:
            return _extract_numeric_score(match.group(1))
    return None


def _split_numbered_blocks(text: str) -> dict[str, list[str]] | None:
    """Partition on numbered headers like ``1. Text Faithfulness``.

    Returns ``None`` when no numbered header is found so the caller can fall
    back to bare-header splitting.
    """
    blocks: dict[str, list[str]] = {key: [] for key in SUPPORTED_ASPECTS}
    current: str | None = None
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        for key, (prefix, header) in SUPPORTED_ASPECTS.items():
            if stripped.startswith(prefix) and header in stripped:
                current = key
                found = True
                break
        if current is not None:
            blocks[current].append(line)
    return blocks if found else None


def _split_header_blocks(text: str) -> dict[str, list[str]]:
    """Partition on bare aspect headers like ``Text Faithfulness:``."""
    blocks: dict[str, list[str]] = {key: [] for key in SUPPORTED_ASPECTS}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        for key, (_, header) in SUPPORTED_ASPECTS.items():
            if stripped.startswith(f"{header}:"):
                current = key
                break
        if current is not None:
            blocks[current].append(line)
    return blocks


def parse_scores_from_detailed_judgement(text: str) -> dict[str, float | str | None]:
    """Split the judge response into per-aspect blocks and extract each score.

    Returns a dict mapping every supported aspect key to a float score, the
    ``"N/A"`` sentinel, or ``None`` when the aspect block is missing.
    """
    # Drop the summary/refinement tail so it cannot pollute score parsing.
    summary_idx = text.find("# Summary:")
    if summary_idx != -1:
        text = text[:summary_idx]
    blocks = _split_numbered_blocks(text)
    if blocks is None:
        blocks = _split_header_blocks(text)
    return {key: _extract_score_from_block(lines) for key, lines in blocks.items()}


class RationalRewardsT2IReward(BaseReward):
    """VLM-judge reward (``rational_rewards_t2i``).

    Async-only: the synchronous ``_score`` raises ``NotImplementedError``.  The
    primary component is the ``N/A``-aware mean of the configured ``aspects``
    mapped to ``[0, 1]``; per-aspect sub-scores are exposed as zero-weight
    components for logging when :attr:`expose_aspects` is set.
    """

    type: Literal["rational_rewards_t2i"] = "rational_rewards_t2i"
    llm: LLMClient

    aspects: list[str] = [
        "text_faithfulness",
        "physical_quality",
        "text_rendering",
    ]
    """Which rubric aspects participate in the averaged reward."""
    expose_aspects: bool = True
    """Expose per-aspect sub-scores as zero-weight components for logging."""
    max_retries: int = 5
    """Retries on transport errors before falling back to a score of 0.0."""

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, _context: Any) -> None:
        unknown = [a for a in self.aspects if a not in SUPPORTED_ASPECTS]
        if unknown:
            raise ValueError(
                f"Unknown rational_rewards aspects {unknown}; "
                f"supported: {sorted(SUPPORTED_ASPECTS)}"
            )
        if not self.aspects:
            raise ValueError("rational_rewards requires at least one aspect.")

    @property
    def component_weights(self) -> list[float]:
        if self.expose_aspects:
            return [1.0, *([0.0] * len(self.aspects))]
        return [1.0]

    @property
    def component_labels(self) -> list[str]:
        if self.expose_aspects:
            return [self.type, *self.aspects]
        return [self.type]

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _load_model(self, device: torch.device) -> None:
        pass

    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError(
            "RationalRewardsT2IReward is async-only. Use async_score() instead."
        )

    def _build_user_content(
        self, prompt: str, image: torch.Tensor
    ) -> list[TextContent | ImageContent]:
        # Match Flow-Factory exactly: the image is sandwiched between the
        # ``text_before`` preamble and the rubric suffix.
        text_before = (
            f"User Instruction: {prompt}\n"
            "You are provided with one image:\n"
            "1. Generated Image "
        )
        return [
            {"type": "text", "text": text_before},
            {"type": "image_url", "image_url": {"url": self.llm.encode_image(image)}},
            {"type": "text", "text": T2I_SCORING_PROMPT_SUFFIX},
        ]

    def _aspect_unit_score(self, value: float | str | None) -> float | None:
        """Map a raw [1, 4] aspect score to [0, 1]; ``None`` if N/A / missing."""
        if value is None or value == "N/A":
            return None
        score = max(1.0, min(4.0, float(value)))
        return (score - 1.0) / 3.0

    def _zero_scores(self) -> torch.Tensor:
        return torch.zeros(len(self.component_weights), dtype=torch.float32)

    async def _async_score(self, batch: dict[str, Any]) -> torch.Tensor:
        prompt_text: str = batch["prompt"]
        image: torch.Tensor = batch["clean_image"]
        user_content = self._build_user_content(prompt_text, image)

        content: str | None = None
        for attempt in range(self.max_retries):
            try:
                content, _ = await self.llm.generate(
                    user_content,
                    system_prompt=RATIONAL_T2I_SYSTEM_PROMPT,
                    strip_think=False,
                )
                break
            except Exception as exc:  # noqa: BLE001 - retry any transport failure
                if attempt + 1 >= self.max_retries:
                    logger.warning(
                        "rational_rewards judge failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )
                    return self._zero_scores()
                await asyncio.sleep(min(2.0**attempt, 30.0))

        if not content:
            logger.warning("rational_rewards judge returned empty content.")
            return self._zero_scores()

        try:
            parsed = parse_scores_from_detailed_judgement(content)
        except ValueError as exc:
            logger.warning(
                "rational_rewards failed to parse judge response (%s): %s",
                exc,
                content[:200],
            )
            return self._zero_scores()

        unit_scores = {a: self._aspect_unit_score(parsed.get(a)) for a in self.aspects}
        usable = [s for s in unit_scores.values() if s is not None]
        overall = sum(usable) / len(usable) if usable else 0.0

        values = [overall]
        if self.expose_aspects:
            values.extend(s if s is not None else 0.0 for s in unit_scores.values())
        return torch.tensor(values, dtype=torch.float32)

    def supports_rollout_overlap(self) -> bool:
        return True


if __name__ == "__main__":
    from rich import print

    sample_response = """# User Request Analysis
The user wants a red cube on a wooden table.
# Detailed Judgement
1. Text Faithfulness:
## Justification: The cube is red and on a table.
## Score: 3.5
2. Physical and Visual Quality:
## Justification: Lighting looks plausible.
## Score: 4.0
3. Text Rendering:
## Justification: No text required.
## Score: N/A
# Summary: Good generation. Score: 9.9 should be ignored.

# User Request Refinement:
## Refined Request: A glossy red cube on an oak table.
"""
    parsed = parse_scores_from_detailed_judgement(sample_response)
    print("parsed:", parsed)
    assert parsed["text_faithfulness"] == 3.5, parsed
    assert parsed["physical_quality"] == 4.0, parsed
    assert parsed["text_rendering"] == "N/A", parsed

    reward = RationalRewardsT2IReward(llm=LLMClient())
    print("component_labels:", reward.component_labels)
    print("component_weights:", reward.component_weights)
    # N/A-aware mean of (3.5, 4.0) -> 3.75 -> (3.75-1)/3 = 0.9167
    units = {a: reward._aspect_unit_score(parsed.get(a)) for a in reward.aspects}
    print("unit scores:", units)
    usable = [s for s in units.values() if s is not None]
    expected_overall = sum(usable) / len(usable)
    print("expected overall:", expected_overall)
    # N/A-aware mean of (3.5, 4.0) -> 3.75 -> (3.75-1)/3.
    assert abs(expected_overall - ((3.75 - 1.0) / 3.0)) < 1e-6
    assert units["text_rendering"] is None
    print("[green]rational_rewards self-test passed[/green]")
