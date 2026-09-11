"""Batch-driven tiling of model evaluation; solver state stays whole-image."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.distributed as dist
from einops import rearrange

from flow_control.adapters.base import SamplerModel
from flow_control.processors.tiles import TileConfig


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


def _window(size: tuple[int, int], blend: str, device: torch.device) -> torch.Tensor:
    axes = []
    for length in size:
        # Cell centres keep Hann strictly positive at uncovered image boundaries.
        u = (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length
        if blend == "hann":
            weight = torch.sin(torch.pi * u).square()
        elif blend == "gaussian":
            weight = torch.exp(-0.5 * ((u - 0.5) / 0.25).square())
        else:
            weight = torch.ones_like(u)
        axes.append(weight)
    return axes[0][None, :, None, None] * axes[1][None, None, :, None]


@dataclass
class _Tiles:
    batches: list[Any]
    origins: list[tuple[int, int]]
    grid: torch.Tensor
    window: torch.Tensor

    def combine(self, velocities: list[torch.Tensor]) -> torch.Tensor:
        output = torch.zeros_like(self.grid, dtype=torch.float32)
        weight = torch.zeros_like(self.grid[..., :1], dtype=torch.float32)
        h, w = self.window.shape[1:3]
        for (top, left), velocity in zip(self.origins, velocities, strict=True):
            output[:, top : top + h, left : left + w] += (
                rearrange(velocity.float(), "b (h w) d -> b h w d", h=h, w=w)
                * self.window
            )
            weight[:, top : top + h, left : left + w] += self.window
        return rearrange(output / weight, "b h w d -> b (h w) d")


def _prepare_tiles(model: SamplerModel, batch: dict[str, Any]) -> _Tiles:
    if not all(
        hasattr(model, name)
        for name in ("patch_size", "vae_scale_factor", "prepare_tile_batch")
    ):
        raise ValueError(
            "Tiled evaluation requires an adapter with packed image geometry and prepare_tile_batch."
        )
    leaf = cast(TileModel, model)
    config = TileConfig.model_validate(batch["tiling"])
    scale = leaf.patch_size * leaf.vae_scale_factor
    config.validate_stride(scale)
    height, width = batch["image_size"]
    x = batch["noisy_latents"]
    if (
        height % scale
        or width % scale
        or x.ndim != 3
        or x.shape[:2] != (1, height * width // scale**2)
    ):
        raise ValueError(
            f"Tiled evaluation requires one packed BND image matching image_size; "
            f"got {tuple(x.shape)}, image_size={(height, width)}, stride={scale}."
        )
    grid = rearrange(x, "b (h w) d -> b h w d", h=height // scale, w=width // scale)
    tile_h, tile_w = config.size_for((height, width))
    origins = config.origins((height, width))
    conditions = batch.get("tiles")
    if conditions is not None and len(conditions) != len(origins):
        raise ValueError(
            f"Expected {len(origins)} row-major tile batches, got {len(conditions)}."
        )
    tiles = []
    for index, (top, left) in enumerate(origins):
        source = batch if conditions is None else conditions[index]
        if conditions is not None and tuple(source["image_size"]) != (tile_h, tile_w):
            raise ValueError(f"Tile {index} image_size must be {(tile_h, tile_w)}.")
        tile = leaf.prepare_tile_batch(
            source, (top, left), (tile_h, tile_w), config.position
        )
        tile["noisy_latents"] = rearrange(
            grid[
                :,
                top // scale : (top + tile_h) // scale,
                left // scale : (left + tile_w) // scale,
            ],
            "b h w d -> b (h w) d",
        )
        tiles.append(tile)
    return _Tiles(
        tiles,
        [(top // scale, left // scale) for top, left in origins],
        grid,
        _window((tile_h // scale, tile_w // scale), config.blend, model.device),
    )


@dataclass
class TiledModel:
    """Read each batch's processor-produced tiling; pass ordinary batches through."""

    model: SamplerModel

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
        prepared = [
            _prepare_tiles(self.model, batch) if "tiling" in batch else None
            for batch in batches
        ]
        tiles, times = [], []
        for batch, timestep, item in zip(batches, timesteps, prepared, strict=True):
            expanded = item.batches if item is not None else [batch]
            tiles.extend(expanded)
            times.extend([timestep] * len(expanded))
        real_count = len(tiles)
        if dist.is_initialized():
            # Every rank participates, including ranks with only ordinary batches.
            count = torch.tensor(real_count, device=self.device, dtype=torch.int64)
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            padding = int(count.item()) - real_count
            times.extend([times[-1]] * padding)
            tiles.extend([tiles[-1]] * padding)
        velocities = self.model.predict_velocity_batched(tiles, times)
        outputs = []
        offset = 0
        for item in prepared:
            count = len(item.batches) if item is not None else 1
            values = velocities[offset : offset + count]
            outputs.append(item.combine(values) if item is not None else values[0])
            offset += count
        if torch.is_grad_enabled() and len(velocities) > real_count:
            # Discarded sequential forwards still need their backward collectives.
            outputs[0] = outputs[0] + sum(
                velocity.sum() * 0.0 for velocity in velocities[real_count:]
            )
        return outputs
