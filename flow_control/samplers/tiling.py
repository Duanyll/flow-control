"""Tile-local prediction and whole-image reconstruction in one coroutine."""

from dataclasses import replace
from typing import Any, Literal, cast

import torch
from einops import rearrange

from flow_control.adapters.base import Batch
from flow_control.utils.tiling import TileLayout, extract_tiles, stitch_tiles

from .calls import Calls, gather
from .plan import EvalRequest, StepContext
from .prediction import Predictor, WrappedPrediction, prediction_registry


@prediction_registry.register("tiled")
class TiledPrediction(WrappedPrediction):
    """Bind a separate child per tile; absent layout passes through unchanged.

    Conditions/layout are fixed by the processor. Each evaluation cuts the
    current whole-image latent, evaluates children together, then stitches.
    """

    type: Literal["tiled"] = "tiled"

    def bind(self, row: Batch, negative_row: Batch | None = None) -> Predictor:
        source = cast(dict[str, Any], row)
        if "tiling" not in source:
            return self.inner.bind(row, negative_row)
        layout = TileLayout.model_validate(source["tiling"])
        specs = layout.token_specs(row["image_size"])
        height, width = (length // layout.stride for length in row["image_size"])

        def condition(original: Batch | None, index: int) -> Batch | None:
            if original is None:
                return None
            data = cast(dict[str, Any], original)
            tiles = data.get("tiles")
            if tiles is not None and len(tiles) != len(specs):
                raise ValueError(
                    f"Expected {len(specs)} tile conditions, got {len(tiles)}."
                )
            tile = dict(data if tiles is None else tiles[index])
            for key in (
                "tiling",
                "tiles",
                "model_image_size",
                "negative",
                "clean_latents",
            ):
                tile.pop(key, None)
            spec = specs[index]
            tile["image_size"] = (
                spec.height * layout.stride,
                spec.width * layout.stride,
            )
            return cast(Batch, tile)

        children = [
            self.inner.bind(cast(Batch, condition(row, i)), condition(negative_row, i))
            for i in range(len(specs))
        ]

        def predict(request: EvalRequest, ctx: StepContext) -> Calls[torch.Tensor]:
            x = request.latents
            if x.ndim != 3 or x.shape[:2] != (1, height * width):
                raise ValueError(
                    "Tiled prediction requires one packed BND image matching image_size; "
                    f"got {tuple(x.shape)}, image_size={row['image_size']}, stride={layout.stride}."
                )
            grid = rearrange(x, "b (h w) d -> b d h w", h=height, w=width)
            velocities = yield from gather(
                [
                    child(
                        replace(
                            request, latents=rearrange(tile, "b d h w -> b (h w) d")
                        ),
                        ctx,
                    )
                    for child, tile in zip(
                        children, extract_tiles(grid, specs), strict=True
                    )
                ]
            )
            tiles = [
                rearrange(v, "b (h w) d -> b d h w", h=spec.height, w=spec.width)
                for v, spec in zip(velocities, specs, strict=True)
            ]
            stitched = stitch_tiles(tiles, specs, height, width)
            return rearrange(stitched, "b d h w -> b (h w) d")

        return predict
