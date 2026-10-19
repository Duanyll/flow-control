"""Prediction nodes: a velocity at the row's point; the row does not move."""

from __future__ import annotations

import math
from typing import Annotated, Literal, cast

import torch
from einops import rearrange
from pydantic import BaseModel, SerializeAsAny

from flow_control.utils.registry import Registry, RegistryUnion
from flow_control.utils.tiling import TileLayout, TileSpec, extract_tiles, stitch_tiles

from .calls import Calls, ModelCall, gather
from .node import BaseNode, Frame, Row, caller, data, derived, follow, walk


class BasePrediction(BaseNode):
    def __call__(self, row: Row) -> Calls[torch.Tensor]:
        with self.enter(row) as frame:
            return (yield from self.run(row, frame))

    def run(self, row: Row, frame: Frame) -> Calls[torch.Tensor]:
        raise NotImplementedError


prediction_registry: Registry[BasePrediction] = Registry(
    "prediction", base=BasePrediction
)
Prediction = SerializeAsAny[
    Annotated[
        BasePrediction,
        RegistryUnion(prediction_registry, "type", number_as=("cfg", "scale")),
    ]
]


@prediction_registry.register("model")
class Model(BasePrediction):
    type: Literal["model"] = "model"
    variant: str | None = None
    """A model variant (model-variants.md); None is the engine's."""

    def run(self, row: Row, frame: Frame) -> Calls[torch.Tensor]:
        if "tiling" in row:
            raise ValueError(
                "The row carries a tile layout but no `tiled` node wraps this model "
                'prediction; tiling is opt-in: add {"type": "tiled"} around the model.'
            )
        (velocity,) = yield [ModelCall(row, self.variant)]
        return velocity

    def estimate_calls(self, row: Row) -> int:
        return 1


def variants(tree: BaseModel) -> list[str | None]:
    """The model variants a config tree asks for; the same order on every rank."""
    return list(
        dict.fromkeys(node.variant for node in walk(tree) if isinstance(node, Model))
    )


@prediction_registry.register("cfg")
class Cfg(BasePrediction):
    """Classifier-free guidance between the conditional and a negative branch."""

    type: Literal["cfg"] = "cfg"
    scale: float = 1.0
    prediction: Prediction = Model()
    negative: Prediction | None = None
    """The negative branch; defaults to `prediction`."""
    negative_condition: Literal["negative", "positive"] = "negative"
    """``negative`` evaluates the row's negative conditioning, ``positive`` the
    row itself (autoguidance between variants)."""
    renorm: bool = False
    renorm_eps: float = 1e-8
    renorm_min: float = 0.0
    renorm_max: float | None = 1.0
    """Upper bound of the per-token renorm factor ``|cond| / |guided|``; the
    default only shrinks tokens, ``None`` rescales every token to the conditional norm."""

    def guided(self) -> bool:
        return self.scale != 1.0

    def run(self, row: Row, frame: Frame) -> Calls[torch.Tensor]:
        if not self.guided():
            return (yield from self.prediction(row))
        uncond = follow(derived(row, "_uncond", lambda: self.uncond_row(row)), row)
        cond, neg = yield from gather(
            [self.prediction(row), (self.negative or self.prediction)(uncond)]
        )
        return self.guide(row, cond, neg)

    def uncond_row(self, row: Row) -> Row:
        sub = data(row)
        if self.negative_condition == "negative":
            if "negative" not in sub:
                raise ValueError(
                    f"{self.type} needs the row's negative conditioning; "
                    "set processor.save_negative."
                )
            sub.update(sub.pop("negative"))
        return sub

    def guide(self, row: Row, cond: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        combined = neg + (cond - neg) * self.scale
        if self.renorm:
            cond_norm = torch.norm(cond, dim=2, keepdim=True)
            noise_norm = torch.norm(combined, dim=2, keepdim=True)
            combined = combined * (cond_norm / (noise_norm + self.renorm_eps)).clamp(
                min=self.renorm_min, max=self.renorm_max
            )
        return combined

    def estimate_calls(self, row: Row) -> int:
        cond = self.prediction.estimate_calls(row)
        if not self.guided():
            return cond
        return cond + (self.negative or self.prediction).estimate_calls(row)


def kappa(sigma: float, sigma_next: float, eta: float, ddim: bool) -> float:
    """CFG++'s mean-preserving factor for a flow (or DDIM) step."""
    if sigma == 1.0:
        return 1.0
    numerator = sigma * (1 - sigma_next)
    if not ddim:
        return numerator / ((1 + eta**2 / 2) * (sigma - sigma_next))
    noise_squared = (eta * sigma_next) ** 2 * max(
        1 - sigma_next**2 * (1 - sigma) ** 2 / numerator**2, 0.0
    )
    sigma_down = math.sqrt(max(sigma_next**2 - noise_squared, 0.0))
    return numerator / (numerator - (1 - sigma) * sigma_down)


@prediction_registry.register("cfg_pp")
class CfgPlusPlus(Cfg):
    """First-order CFG++: guide, then preserve the enclosing step's mean."""

    type: Literal["cfg_pp"] = "cfg_pp"

    def guided(self) -> bool:
        return True

    def guide(self, row: Row, cond: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        from .transition import Ddim, Euler, StepFrame

        step = cast(
            StepFrame,
            caller(
                row, (Euler, Ddim), who="cfg_pp needs an enclosing euler or ddim step"
            ),
        )
        node = cast(Euler | Ddim, step.node)
        factor = kappa(step.sigma, step.sigma_next, node.eta, isinstance(node, Ddim))
        return neg + factor * (super().guide(row, cond, neg) - neg)


def tile_specs(row: Row) -> tuple[int, list[TileSpec]]:
    """The row's packed-token tile plan: ``(stride, specs)``."""
    layout = TileLayout.model_validate(row["tiling"])
    return layout.stride, layout.token_specs(row["image_size"])


def tile_rows(row: Row, specs: list[TileSpec], stride: int) -> list[Row]:
    """One conditioning row per tile, from ``row["tiles"]`` or the row itself."""
    rows = []
    for source, spec in zip(row.get("tiles") or [row] * len(specs), specs, strict=True):
        tile = data(source)
        for key in ("tiling", "tiles", "model_image_size", "clean_latents"):
            tile.pop(key, None)
        tile["image_size"] = (spec.height * stride, spec.width * stride)
        rows.append(tile)
    return rows


@prediction_registry.register("tiled")
class Tiled(BasePrediction):
    """Predict each tile of a tiled row on its own, then feather-stitch."""

    type: Literal["tiled"] = "tiled"
    prediction: Prediction = Model()

    def run(self, row: Row, frame: Frame) -> Calls[torch.Tensor]:
        if "tiling" not in row:
            return (yield from self.prediction(row))
        stride, specs = tile_specs(row)
        height, width = (length // stride for length in row["image_size"])
        grid = rearrange(
            row["noisy_latents"], "b (h w) d -> b d h w", h=height, w=width
        )
        tiles = derived(row, "_tiles", lambda: tile_rows(row, specs, stride))
        for tile, crop in zip(tiles, extract_tiles(grid, specs), strict=True):
            follow(tile, row)["noisy_latents"] = rearrange(crop, "b d h w -> b (h w) d")
        velocities = yield from gather([self.prediction(tile) for tile in tiles])
        stitched = stitch_tiles(
            [
                rearrange(v, "b (h w) d -> b d h w", h=spec.height, w=spec.width)
                for v, spec in zip(velocities, specs, strict=True)
            ],
            specs,
            height,
            width,
        )
        return rearrange(stitched, "b d h w -> b (h w) d")

    def estimate_calls(self, row: Row) -> int:
        count = len(tile_specs(row)[1]) if "tiling" in row else 1
        return count * self.prediction.estimate_calls(row)
