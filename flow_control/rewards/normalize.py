"""Configurable score normalization for rewards.

Each ``BaseReward._score`` returns the raw, model-native score.  The
``BaseReward.normalize`` field (a discriminated union defined here) then
post-processes that raw score before it is exposed to trainers / loggers.

The transforms are deliberately element-wise so they compose cleanly with
multi-component rewards (``[C]``-shaped tensors).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter


class _NormalizeBase(BaseModel):
    """Common base for the normalize variants."""

    model_config = ConfigDict(extra="forbid")

    def apply(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - overridden
        raise NotImplementedError


class IdentityNormalize(_NormalizeBase):
    """No-op normalize (default).  Raw score is returned unchanged."""

    type: Literal["identity"] = "identity"

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AffineNormalize(_NormalizeBase):
    """``x * scale + offset``.

    Use for fixed linear rescaling, e.g. dividing CLIP logits by 30 or
    PickScore logits by 26 to roughly land in ``[0, 1]``.
    """

    type: Literal["affine"] = "affine"
    scale: float = 1.0
    offset: float = 0.0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset


class SigmoidNormalize(_NormalizeBase):
    """``sigmoid((x - offset) * scale)``.

    Use for squashing unbounded scores (e.g. ImageReward's normalized output)
    into ``(0, 1)``.  ``offset`` shifts the inflection point; ``scale``
    controls steepness.
    """

    type: Literal["sigmoid"] = "sigmoid"
    scale: float = 1.0
    offset: float = 0.0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.offset) * self.scale)


class ClampNormalize(_NormalizeBase):
    """``x.clamp(low, high)``.

    Useful as a safety net after an affine transform to keep rewards in a
    well-defined range.
    """

    type: Literal["clamp"] = "clamp"
    low: float = 0.0
    high: float = 1.0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(self.low, self.high)


Normalize = Annotated[
    Annotated[IdentityNormalize, Tag("identity")]
    | Annotated[AffineNormalize, Tag("affine")]
    | Annotated[SigmoidNormalize, Tag("sigmoid")]
    | Annotated[ClampNormalize, Tag("clamp")],
    Discriminator("type"),
]


_normalize_ta = TypeAdapter(Normalize)


def parse_normalize(conf: dict[str, Any]) -> _NormalizeBase:
    """Parse a normalize config dict into the appropriate instance."""
    return _normalize_ta.validate_python(conf)


__all__ = [
    "AffineNormalize",
    "ClampNormalize",
    "IdentityNormalize",
    "Normalize",
    "SigmoidNormalize",
    "parse_normalize",
]


if __name__ == "__main__":
    from rich import print

    x = torch.tensor([0.0, 1.0, 2.0])
    print("input:", x.tolist())
    print("identity:", IdentityNormalize().apply(x).tolist())
    print("affine(0.5, +1):", AffineNormalize(scale=0.5, offset=1.0).apply(x).tolist())
    print("sigmoid(1, 1):", SigmoidNormalize(scale=1.0, offset=1.0).apply(x).tolist())
    print("clamp(0, 1.5):", ClampNormalize(low=0.0, high=1.5).apply(x).tolist())

    # Round-trip through TypeAdapter to confirm discriminator works.
    parsed = parse_normalize({"type": "sigmoid", "scale": 2.0, "offset": 1.5})
    print("parsed:", parsed)
    print("parsed.apply:", parsed.apply(x).tolist())
