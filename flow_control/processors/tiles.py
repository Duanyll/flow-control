"""Shared pixel-space tile layout stored in processed batches."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class TileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_size: tuple[int, int] = (1024, 1024)
    overlap: tuple[int, int] = (128, 128)
    position: Literal["local", "global"] = "local"
    blend: Literal["uniform", "gaussian", "hann"] = "uniform"

    @model_validator(mode="after")
    def _validate_geometry(self) -> "TileConfig":
        if any(
            size <= 0 or not 0 <= overlap < size
            for size, overlap in zip(self.tile_size, self.overlap, strict=True)
        ):
            raise ValueError(
                "Each tile dimension must be positive with 0 <= overlap < tile_size."
            )
        return self

    def validate_stride(self, stride: int) -> None:
        if any(value % stride for value in (*self.tile_size, *self.overlap)):
            raise ValueError(
                f"tile_size and overlap must be multiples of the packed pixel stride ({stride})."
            )

    def size_for(self, image_size: tuple[int, int]) -> tuple[int, int]:
        if min(image_size) <= 0:
            raise ValueError(
                f"Tiled image dimensions must be positive, got {image_size}."
            )
        return min(image_size[0], self.tile_size[0]), min(
            image_size[1], self.tile_size[1]
        )

    def origins(self, image_size: tuple[int, int]) -> list[tuple[int, int]]:
        """Return row-major pixel origins, anchoring the final tile to each edge."""
        size = self.size_for(image_size)
        axes: list[list[int]] = []
        for length, tile, overlap in zip(image_size, size, self.overlap, strict=True):
            end = length - tile
            # A small image uses one full-image tile, irrespective of overlap.
            positions = list(range(0, end + 1, tile - overlap)) if end else [0]
            if positions[-1] != end:
                positions.append(end)
            axes.append(positions)
        return [(top, left) for top in axes[0] for left in axes[1]]
