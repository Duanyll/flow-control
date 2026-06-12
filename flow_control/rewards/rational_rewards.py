"""RationalRewards VLM-judge rewards — API-based reasoning-reward models.

This ports the ``rational_rewards_t2i`` and ``rational_rewards_edit`` rewards from
Flow-Factory.  Each calls an OpenAI-compatible vLLM server hosting a
``RationalRewards-8B`` judge model, which scores a generated/edited image on a
rubric of aspects (1-4 scale) and emits a structured text response.  The aspect
scores are parsed out, averaged (``N/A``-aware), and mapped to ``[0, 1]``.

Two variants are provided:

- :class:`RationalRewardsT2IReward` (``rational_rewards_t2i``) judges a generated
  image against the prompt over three aspects.
- :class:`RationalRewardsEditReward` (``rational_rewards_edit``) judges an *edited*
  image against the source image (``reference_images[0]``) and the edit
  instruction over four aspects (adds *Image Faithfulness*).

For both, the aggregated score is the primary reward component (drives the
advantage); the individual aspect sub-scores are exposed as additional
*zero-weight* components so they show up in ``rollout/raw/<aspect>_*`` /
``rollout/normalized/<aspect>_*`` metrics without affecting optimization.
"""

import asyncio
import re
from typing import Any, ClassVar, Literal

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

# Edit rubric inserts "Image Faithfulness" as aspect 2, pushing the shared
# aspects down by one (matches Flow-Factory ``rational_rewards_edit``).
SUPPORTED_EDIT_ASPECTS: dict[str, tuple[str, str]] = {
    "text_faithfulness": ("1.", "Text Faithfulness"),
    "image_faithfulness": ("2.", "Image Faithfulness"),
    "physical_quality": ("3.", "Physical and Visual Quality"),
    "text_rendering": ("4.", "Text Rendering"),
}

RATIONAL_T2I_SYSTEM_PROMPT = (
    "You are an expert image generation evaluator. Your task is to evaluate the "
    "quality of a generated image based on a user instruction. Afterwards, you "
    "need to suggest how to refine the original user request to produce better "
    "image generation (if any)."
)

RATIONAL_EDIT_SYSTEM_PROMPT = (
    "You are an expert image editing evaluator. Your task is to evaluate the "
    "quality of an edited image based on a source image and a user instruction. "
    "Afterwards, you need to suggest how to refine the original user request to "
    "produce better image edits (if any)."
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

# Verbatim edit rubric from Flow-Factory ``rational_rewards_edit`` (its
# ``COMMON_TASK_GUIDELINE``).  The "Give your analysis..." preamble is prepended
# in :meth:`RationalRewardsEditReward._build_user_content`, matching Flow-Factory.
EDIT_SCORING_PROMPT_SUFFIX = """To do this, you must first assess the image on four critical aspects, provide justifications and absolute scores in 1-4 scale.

### Critical Aspects & Scoring Rubric
**1. Text Faithfulness** (How accurately does the output follow the instruction?)
- **4 (Full match):** All key elements (objects, colors, actions) are represented exactly as described. No hallucinations or unrequested changes.
- **3 (Minor mismatch):** Most key elements are present, but minor details are missing, incorrect, or slightly inaccurate.
- **2 (Some mismatch):** Some key elements are missing, altered, or interpreted incorrectly.
- **1 (Major deviations):** Key elements are completely missing, altered, or contradicted. Instruction is ignored.

**2. Image Faithfulness** (How well are the non-edited parts and key input elements preserved?)
- **4 (Uses input fully):** All relevant elements from the input (background, style, lighting, identity) are accurately preserved or transformed as instructed.
- **3 (Minor mismatch):** Most relevant elements are preserved, but a few aspects (e.g., background details, lighting consistency) are missing or incorrectly handled.
- **2 (Partial mismatch):** Some elements are carried over, but key aspects of the original image are lost or distorted.
- **1 (Fails to use input):** Key elements of the input image are ignored, misinterpreted, or destroyed.

**3. Physical and Visual Quality** (Technical errors, composition, realism, and physics)
- **4 (No noticeable flaws):** The image is physically plausible (correct lighting, shadows, geometry, anatomy). No visible artifacts (seams, blurring, noise).
- **3 (Minor flaws):** Small inaccuracies that are noticeable but not strongly disruptive (e.g., slight lighting mismatch, minor texture issues).
- **2 (Some flaws):** Clear physical or visual errors that disrupt the image (e.g., incorrect perspective, "floating" objects, wrong shadow direction, obvious seams).
- **1 (Severe flaws):** Major physical/visual errors (e.g., impossible geometry, distorted anatomy, garbled objects, severe artifacts).

**4. Text Rendering** (Only if the instruction involves generating text)
- **4 (Full match):** Text is correct, legible, and integrated well.
- **3 (Mostly match):** Minor misspellings or inconsistent capitalization.
- **2 (Partial match):** Major misspellings or distorted text.
- **1 (Major deviations):** Text is unreadable, severely distorted, or missing. (Use N/A if no text generation is required).

### Scoring Methodology (CRITICAL)
During assessment for each aspect, recall the initial user request, source image and the scoring rubrics of the aspect, provide scores with detailed justifications for each image and reflect fine-grained preferences.
1. **Anchor:** Have a global inspection based on the user request and the resulting generation. Determine the rough integer score level (1, 2, 3, or 4) according to the definitions provided .
2. **Justify and Adjust:** Do careful visual analysis and identify specific flaws in generation. Justify the score with concrete evidence and scoring logic. Fine-tune this anchor score into a float value. Add small increments for exceptional execution or deduct points for specific flaws.
   - *Example:* deduct points from 4.0 for slight flaws if the assessed dimension is close to satisfaction. add increments from 1.0 or 2.0 based on severity of flaws.

Afterwards, try to construct a refined user request that helps the visual generation model to produce better image edits.
Think of the weaknesses identified in the judgement, then map them to instruction details and apply specific fixes.
Provide a final new user request that enrich the initial user request.

Output your evaluation in the following format:
# User Request Analysis
[ understanding the user request, try to analyze or decompose the user request deeper. Think of what the request might imply or what needs to be inferred to successfully execute the request. ]
# Detailed Judgement
1. Text Faithfulness:
## Justification: [ Analysis of the user request and the assessment of the resulting generation. How it comes to a final score. ]
## Score: [ float score ]
2. Image Faithfulness:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score ]
3. Physical and Visual Quality:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score ]
4. Text Rendering:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score or N/A ]
# Summary: [ Summary of the evaluation ]

# User Request Refinement:
## Refinement Comments: [Specific suggestions for improving the user request]
## Refined Request: [The improved, more specific user request for editing like a standard user instruction]"""

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


def _split_numbered_blocks(
    text: str, aspects: dict[str, tuple[str, str]]
) -> dict[str, list[str]] | None:
    """Partition on numbered headers like ``1. Text Faithfulness``.

    Returns ``None`` when no numbered header is found so the caller can fall
    back to bare-header splitting.
    """
    blocks: dict[str, list[str]] = {key: [] for key in aspects}
    current: str | None = None
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        for key, (prefix, header) in aspects.items():
            if stripped.startswith(prefix) and header in stripped:
                current = key
                found = True
                break
        if current is not None:
            blocks[current].append(line)
    return blocks if found else None


def _split_header_blocks(
    text: str, aspects: dict[str, tuple[str, str]]
) -> dict[str, list[str]]:
    """Partition on bare aspect headers like ``Text Faithfulness:``."""
    blocks: dict[str, list[str]] = {key: [] for key in aspects}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        for key, (_, header) in aspects.items():
            if stripped.startswith(f"{header}:"):
                current = key
                break
        if current is not None:
            blocks[current].append(line)
    return blocks


def parse_scores_from_detailed_judgement(
    text: str, aspects: dict[str, tuple[str, str]] = SUPPORTED_ASPECTS
) -> dict[str, float | str | None]:
    """Split the judge response into per-aspect blocks and extract each score.

    Returns a dict mapping every supported aspect key to a float score, the
    ``"N/A"`` sentinel, or ``None`` when the aspect block is missing.
    """
    # Drop the summary/refinement tail so it cannot pollute score parsing.
    summary_idx = text.find("# Summary:")
    if summary_idx != -1:
        text = text[:summary_idx]
    blocks = _split_numbered_blocks(text, aspects)
    if blocks is None:
        blocks = _split_header_blocks(text, aspects)
    return {key: _extract_score_from_block(lines) for key, lines in blocks.items()}


class _RationalRewardsReward(BaseReward):
    """Shared logic for the RationalRewards VLM-judge rewards.

    Async-only: the synchronous ``_score`` raises ``NotImplementedError``.  The
    primary component is the ``N/A``-aware mean of the configured ``aspects``
    mapped to ``[0, 1]``; per-aspect sub-scores are exposed as zero-weight
    components for logging when :attr:`expose_aspects` is set.

    Subclasses provide the supported-aspect table, the judge system prompt, the
    consumed batch fields, and the interleaved user content (which images to
    send); the retry / parse / aggregate machinery lives here.
    """

    llm: LLMClient

    aspects: list[str]
    """Which rubric aspects participate in the averaged reward."""
    expose_aspects: bool = True
    """Expose per-aspect sub-scores as zero-weight components for logging."""
    max_retries: int = 5
    """Retries on transport errors before falling back to a score of 0.0."""

    model_config = ConfigDict(extra="forbid")

    # ── Subclass-provided constants ──────────────────────────────────────
    _supported_aspects: ClassVar[dict[str, tuple[str, str]]] = {}
    _system_prompt: ClassVar[str] = ""

    def model_post_init(self, _context: Any) -> None:
        unknown = [a for a in self.aspects if a not in self._supported_aspects]
        if unknown:
            raise ValueError(
                f"Unknown {self.type} aspects {unknown}; "
                f"supported: {sorted(self._supported_aspects)}"
            )
        if not self.aspects:
            raise ValueError(f"{self.type} requires at least one aspect.")

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

    def _load_model(self, device: torch.device) -> None:
        pass

    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} is async-only. Use async_score() instead."
        )

    def _build_user_content(
        self, batch: dict[str, Any]
    ) -> list[TextContent | ImageContent] | None:
        """Build the interleaved judge user content for one sample.

        Returns ``None`` when the sample lacks a required input (e.g. a missing
        source image for editing), signalling :meth:`_async_score` to fall back
        to a zero score.
        """
        raise NotImplementedError

    def _aspect_unit_score(self, value: float | str | None) -> float | None:
        """Map a raw [1, 4] aspect score to [0, 1]; ``None`` if N/A / missing."""
        if value is None or value == "N/A":
            return None
        score = max(1.0, min(4.0, float(value)))
        return (score - 1.0) / 3.0

    def _zero_scores(self) -> torch.Tensor:
        return torch.zeros(len(self.component_weights), dtype=torch.float32)

    async def _async_score(self, batch: dict[str, Any]) -> torch.Tensor:
        user_content = self._build_user_content(batch)
        if user_content is None:
            return self._zero_scores()

        content: str | None = None
        for attempt in range(self.max_retries):
            try:
                content, _ = await self.llm.generate(
                    user_content,
                    system_prompt=self._system_prompt,
                    strip_think=False,
                )
                break
            except Exception as exc:  # noqa: BLE001 - retry any transport failure
                if attempt + 1 >= self.max_retries:
                    logger.warning(
                        "%s judge failed after %d attempts: %s",
                        self.type,
                        self.max_retries,
                        exc,
                    )
                    return self._zero_scores()
                await asyncio.sleep(min(2.0**attempt, 30.0))

        if not content:
            logger.warning("%s judge returned empty content.", self.type)
            return self._zero_scores()

        try:
            parsed = parse_scores_from_detailed_judgement(
                content, self._supported_aspects
            )
        except ValueError as exc:
            logger.warning(
                "%s failed to parse judge response (%s): %s",
                self.type,
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


class RationalRewardsT2IReward(_RationalRewardsReward):
    """VLM-judge reward for text-to-image generation (``rational_rewards_t2i``).

    Scores a generated image against the prompt over three aspects
    (text faithfulness, physical/visual quality, text rendering).
    """

    type: Literal["rational_rewards_t2i"] = "rational_rewards_t2i"

    aspects: list[str] = [
        "text_faithfulness",
        "physical_quality",
        "text_rendering",
    ]

    _supported_aspects: ClassVar[dict[str, tuple[str, str]]] = SUPPORTED_ASPECTS
    _system_prompt: ClassVar[str] = RATIONAL_T2I_SYSTEM_PROMPT

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _build_user_content(
        self, batch: dict[str, Any]
    ) -> list[TextContent | ImageContent]:
        # Match Flow-Factory exactly: the image is sandwiched between the
        # ``text_before`` preamble and the rubric suffix.
        prompt: str = batch["prompt"]
        image: torch.Tensor = batch["clean_image"]
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


class RationalRewardsEditReward(_RationalRewardsReward):
    """VLM-judge reward for instruction-guided image editing (``rational_rewards_edit``).

    Scores an *edited* image (``clean_image``) against the source image
    (``reference_images[0]``) and the edit instruction over four aspects, adding
    *Image Faithfulness* (how well non-edited content is preserved) to the T2I
    rubric.  Falls back to a zero score when no source image is present.
    """

    type: Literal["rational_rewards_edit"] = "rational_rewards_edit"

    aspects: list[str] = [
        "text_faithfulness",
        "image_faithfulness",
        "physical_quality",
        "text_rendering",
    ]

    _supported_aspects: ClassVar[dict[str, tuple[str, str]]] = SUPPORTED_EDIT_ASPECTS
    _system_prompt: ClassVar[str] = RATIONAL_EDIT_SYSTEM_PROMPT

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "prompt", "reference_images"}

    def _source_image(self, batch: dict[str, Any]) -> torch.Tensor | None:
        references = batch.get("reference_images")
        if not references:
            return None
        return references[0]

    def _build_user_content(
        self, batch: dict[str, Any]
    ) -> list[TextContent | ImageContent] | None:
        source = self._source_image(batch)
        if source is None:
            logger.warning(
                "%s got a sample without a source image; scoring 0.0.", self.type
            )
            return None

        prompt: str = batch["prompt"]
        edited: torch.Tensor = batch["clean_image"]
        head = (
            f"User Instruction: {prompt}\n"
            "You are provided with two images:\n"
            "1. Source Image "
        )
        between_images = "\n2. Edited Image "
        after_images = (
            "\n\nGive your analysis and judgement following guidelines in the "
            "system prompt. \n\n" + EDIT_SCORING_PROMPT_SUFFIX
        )
        return [
            {"type": "text", "text": head},
            {"type": "image_url", "image_url": {"url": self.llm.encode_image(source)}},
            {"type": "text", "text": between_images},
            {"type": "image_url", "image_url": {"url": self.llm.encode_image(edited)}},
            {"type": "text", "text": after_images},
        ]


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
    print("parsed (t2i):", parsed)
    assert parsed["text_faithfulness"] == 3.5, parsed
    assert parsed["physical_quality"] == 4.0, parsed
    assert parsed["text_rendering"] == "N/A", parsed

    reward = RationalRewardsT2IReward(llm=LLMClient())
    print("t2i component_labels:", reward.component_labels)
    print("t2i component_weights:", reward.component_weights)
    # N/A-aware mean of (3.5, 4.0) -> 3.75 -> (3.75-1)/3 = 0.9167
    units = {a: reward._aspect_unit_score(parsed.get(a)) for a in reward.aspects}
    print("t2i unit scores:", units)
    usable = [s for s in units.values() if s is not None]
    expected_overall = sum(usable) / len(usable)
    print("t2i expected overall:", expected_overall)
    assert abs(expected_overall - ((3.75 - 1.0) / 3.0)) < 1e-6
    assert units["text_rendering"] is None

    # ── Edit rubric: four aspects, with Image Faithfulness as #2. ──
    edit_response = """# User Request Analysis
Make the sky blue.
# Detailed Judgement
1. Text Faithfulness:
## Justification: The sky is now blue.
## Score: 4.0
2. Image Faithfulness:
## Justification: Foreground preserved well.
## Score: 3.0
3. Physical and Visual Quality:
## Justification: Lighting consistent.
## Score: 3.5
4. Text Rendering:
## Justification: No text involved.
## Score: N/A
# Summary: Solid edit. Score: 7.7 should be ignored.

# User Request Refinement:
## Refined Request: Make the sky a clear cerulean blue.
"""
    edit_parsed = parse_scores_from_detailed_judgement(
        edit_response, SUPPORTED_EDIT_ASPECTS
    )
    print("parsed (edit):", edit_parsed)
    assert edit_parsed["text_faithfulness"] == 4.0, edit_parsed
    assert edit_parsed["image_faithfulness"] == 3.0, edit_parsed
    assert edit_parsed["physical_quality"] == 3.5, edit_parsed
    assert edit_parsed["text_rendering"] == "N/A", edit_parsed

    edit_reward = RationalRewardsEditReward(llm=LLMClient())
    print("edit component_labels:", edit_reward.component_labels)
    print("edit component_weights:", edit_reward.component_weights)
    edit_units = {
        a: edit_reward._aspect_unit_score(edit_parsed.get(a))
        for a in edit_reward.aspects
    }
    print("edit unit scores:", edit_units)
    edit_usable = [s for s in edit_units.values() if s is not None]
    # N/A-aware mean of (4.0, 3.0, 3.5) -> 3.5 -> (3.5-1)/3.
    assert abs(sum(edit_usable) / len(edit_usable) - ((3.5 - 1.0) / 3.0)) < 1e-6
    assert edit_units["text_rendering"] is None
    # Missing source image -> no content -> zero scores.
    assert edit_reward._build_user_content({"prompt": "x", "clean_image": None}) is None

    print("[green]rational_rewards self-test passed[/green]")
