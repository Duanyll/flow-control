"""Square tile layout configured on tiled processors and stored in batches."""

from pydantic import BaseModel, ConfigDict, model_validator

from flow_control.utils.tiling import TileLayout


class TileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_size: int = 1024
    overlap: int = 128

    @model_validator(mode="after")
    def _validate_geometry(self) -> "TileConfig":
        if self.tile_size <= 0 or not 0 <= self.overlap < self.tile_size:
            raise ValueError(
                "tile_size must be positive with 0 <= overlap < tile_size."
            )
        return self

    def validate_stride(self, stride: int) -> None:
        if self.tile_size % stride or self.overlap % stride:
            raise ValueError(
                f"tile_size and overlap must be multiples of the packed pixel stride ({stride})."
            )

    def layout(self, stride: int) -> TileLayout:
        self.validate_stride(stride)
        return TileLayout(tile_size=self.tile_size, overlap=self.overlap, stride=stride)
