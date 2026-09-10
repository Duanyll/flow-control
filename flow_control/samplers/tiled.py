"""Tile only model evaluation; guidance, projection and solver noise stay whole-image."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
import torch.distributed as dist
from einops import rearrange
from pydantic import BaseModel, ConfigDict, model_validator

from flow_control.adapters.base import SamplerModel


class TileModel(SamplerModel, Protocol):
    patch_size: int
    vae_scale_factor: int

    def prepare_tile_batch(
        self,
        batch: dict[str, Any],
        origin: tuple[int, int],
        size: tuple[int, int],
        position: str,
    ) -> dict[str, Any]: ...


class Tiled(BaseModel):
    """Pixel-sized raster tiles with normalized uniform overlap blending.

    Local positions restart each tile's image coordinates. Global positions use
    the full image origin and require explicit support from the model adapter.
    Optional ``batch['tiles']`` supplies one complete conditioning batch per tile,
    in row-major order; each must declare its tile's actual ``image_size``.
    """

    model_config = ConfigDict(extra="forbid")
    tile_size: tuple[int, int] = (1024, 1024)
    overlap: tuple[int, int] = (128, 128)
    position: Literal["local", "global"] = "local"

    @model_validator(mode="after")
    def _validate_geometry(self) -> "Tiled":
        if any(
            size <= 0 or not 0 <= overlap < size
            for size, overlap in zip(self.tile_size, self.overlap, strict=True)
        ):
            raise ValueError(
                "Each tile dimension must be positive with 0 <= overlap < tile_size."
            )
        return self

    def wrap(self, model: SamplerModel) -> SamplerModel:
        if isinstance(model, _TiledModel):
            if model.config != self:
                raise ValueError(
                    "Cannot apply different tiled configurations to the same model."
                )
            return model
        if not all(
            hasattr(model, name)
            for name in ("patch_size", "vae_scale_factor", "prepare_tile_batch")
        ):
            raise ValueError(
                "Tiled sampling requires an adapter with packed image geometry and prepare_tile_batch."
            )
        leaf = cast(TileModel, model)
        scale = leaf.patch_size * leaf.vae_scale_factor
        if any(value % scale for value in (*self.tile_size, *self.overlap)):
            raise ValueError(
                f"tile_size and overlap must be multiples of the packed pixel stride ({scale})."
            )
        return _TiledModel(self, leaf)


def _starts(length: int, size: int, overlap: int) -> list[int]:
    end = max(0, length - size)
    positions = list(range(0, end + 1, size - overlap))
    if positions[-1] != end:
        positions.append(end)
    return positions


@dataclass
class _TiledModel:
    config: Tiled
    model: TileModel

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def use_variant(self, variant: str | None) -> AbstractContextManager[None]:
        return self.model.use_variant(variant)

    def predict_velocity_batched(
        self, batches: list[Any], timesteps: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        if not batches or len(batches) != len(timesteps):
            raise ValueError(
                "Tiled evaluation requires nonempty batches and equally many timesteps."
            )
        scale = self.model.patch_size * self.model.vae_scale_factor
        tiles: list[Any] = []
        times: list[torch.Tensor] = []
        placements: list[tuple[int, int, int, int, int]] = []
        outputs: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for index, (batch, timestep) in enumerate(zip(batches, timesteps, strict=True)):
            height, width = batch["image_size"]
            x = batch["noisy_latents"]
            if (
                height % scale
                or width % scale
                or x.ndim != 3
                or x.shape[:2] != (1, height * width // scale**2)
            ):
                raise ValueError(
                    f"Tiled sampling requires one packed BND image matching image_size; "
                    f"got {tuple(x.shape)}, image_size={(height, width)}, stride={scale}."
                )
            grid = rearrange(
                x, "b (h w) d -> b h w d", h=height // scale, w=width // scale
            )
            outputs.append(
                torch.zeros_like(grid, device=self.device, dtype=torch.float32)
            )
            weights.append(
                torch.zeros((1, height // scale, width // scale, 1), device=self.device)
            )
            tile_h, tile_w = (
                min(height, self.config.tile_size[0]),
                min(width, self.config.tile_size[1]),
            )
            origins = [
                (top, left)
                for top in _starts(
                    height, self.config.tile_size[0], self.config.overlap[0]
                )
                for left in _starts(
                    width, self.config.tile_size[1], self.config.overlap[1]
                )
            ]
            conditions = batch.get("tiles")
            if conditions is not None and len(conditions) != len(origins):
                raise ValueError(
                    f"Expected {len(origins)} row-major tile batches, got {len(conditions)}."
                )
            for tile_index, (top, left) in enumerate(origins):
                source = batch if conditions is None else conditions[tile_index]
                if conditions is not None and tuple(source["image_size"]) != (
                    tile_h,
                    tile_w,
                ):
                    raise ValueError(
                        f"Tile {tile_index} image_size must be {(tile_h, tile_w)}."
                    )
                tile = self.model.prepare_tile_batch(
                    source, (top, left), (tile_h, tile_w), self.config.position
                )
                y, z, h, w = (
                    top // scale,
                    left // scale,
                    tile_h // scale,
                    tile_w // scale,
                )
                tile["noisy_latents"] = rearrange(
                    grid[:, y : y + h, z : z + w], "b h w d -> b (h w) d"
                )
                tiles.append(tile)
                times.append(timestep)
                placements.append((index, y, z, h, w))

        if dist.is_initialized():
            count = torch.tensor(len(tiles), device=self.device, dtype=torch.int64)
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            padding = int(count.item()) - len(tiles)
            times.extend([times[-1]] * padding)
            tiles.extend([tiles[-1]] * padding)
        velocities = self.model.predict_velocity_batched(tiles, times)
        for (index, top, left, height, width), velocity in zip(
            placements, velocities[: len(placements)], strict=True
        ):
            outputs[index][:, top : top + height, left : left + width] += rearrange(
                velocity, "b (h w) d -> b h w d", h=height, w=width
            )
            weights[index][:, top : top + height, left : left + width] += 1
        if torch.is_grad_enabled() and len(velocities) > len(placements):
            # Sequential leaf fallback also needs every padded forward's backward
            # collectives, even though these duplicates contribute no velocity.
            outputs[0] = outputs[0] + sum(
                velocity.sum() * 0.0 for velocity in velocities[len(placements) :]
            )
        return [
            rearrange(value / weight, "b h w d -> b (h w) d")
            for value, weight in zip(outputs, weights, strict=True)
        ]
