"""Overlapping-tile split / feathered-stitch utilities for large images.

Typical use: run a diffusion enhancer on 4K images tile-by-tile (tile size
1024-1536, overlap 128-256), then blend the processed tiles back into a
seamless full-resolution image.

Tile layout (:func:`plan_tiles`), per axis:

- If the image extent is <= ``tile_size``, a single tile spanning the whole
  axis is used (the tile shrinks to the image; no padding, never out of
  bounds).
- Otherwise ``n = ceil((extent - overlap) / (tile_size - overlap))`` tiles of
  exactly ``tile_size`` are placed with their starts evenly spread (rounded to
  integers) over ``[0, extent - tile_size]``.  The first/last tiles are flush
  with the image borders, and the even spacing keeps every adjacent stride
  <= ``tile_size - overlap``, so the actual overlap between neighbours is
  always >= ``overlap``.

The 2D plan is the row-major cartesian product of the two axis plans.

Feathered stitching (:func:`stitch_tiles`) uses a separable raised-cosine
(Hann) window per tile:

- Along each axis the window is flat 1 with a ``sin^2`` ramp on each side that
  has a neighbouring tile; the ramp length equals the neighbours' penetration
  depth into this tile (i.e. the actual overlap for :func:`plan_tiles` grids).
  Sides flush with the image boundary (or without neighbours) stay at 1.
- Ramps are sampled at pixel centres (offset 0.5), so windows are strictly
  positive everywhere.  Hence the accumulated weight is positive wherever at
  least one tile covers a pixel — no division by zero — and in matched overlap
  regions the two ramps are exactly complementary (``sin^2 + cos^2 = 1``),
  giving a smooth constant-power crossfade.
- The output is ``sum(w_i * tile_i) / sum(w_i)``, accumulated in float32 and
  cast back to the input dtype.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from einops import einsum, rearrange

__all__ = ["TileSpec", "extract_tiles", "plan_tiles", "stitch_tiles"]


@dataclass(frozen=True, slots=True)
class TileSpec:
    """Placement of one tile inside the full image, in pixel coordinates."""

    top: int
    left: int
    height: int
    width: int


def _plan_axis(extent: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """Plan one axis; returns ``(start, size)`` pairs (see module docstring)."""
    if extent <= tile_size:
        return [(0, extent)]
    stride = tile_size - overlap
    num_tiles = math.ceil((extent - overlap) / stride)
    span = extent - tile_size
    starts = [round(i * span / (num_tiles - 1)) for i in range(num_tiles)]
    return [(start, tile_size) for start in starts]


def plan_tiles(height: int, width: int, tile_size: int, overlap: int) -> list[TileSpec]:
    """Plan a grid of overlapping tiles covering an ``height x width`` image.

    Args:
        height: Full image height in pixels.
        width: Full image width in pixels.
        tile_size: Target tile edge length (square tiles).
        overlap: Minimum overlap between adjacent tiles, ``0 <= overlap <
            tile_size``.

    Returns:
        Row-major list of :class:`TileSpec` covering every pixel, with edge
        tiles flush against the image borders and adjacent tiles overlapping
        by at least ``overlap`` pixels.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Image size must be positive, got {height}x{width}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0 <= overlap < tile_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < tile_size ({tile_size}), "
            f"got {overlap}"
        )
    return [
        TileSpec(top=top, left=left, height=tile_h, width=tile_w)
        for top, tile_h in _plan_axis(height, tile_size, overlap)
        for left, tile_w in _plan_axis(width, tile_size, overlap)
    ]


def _check_specs_in_bounds(specs: Sequence[TileSpec], height: int, width: int) -> None:
    for spec in specs:
        if (
            spec.top < 0
            or spec.left < 0
            or spec.height < 1
            or spec.width < 1
            or spec.top + spec.height > height
            or spec.left + spec.width > width
        ):
            raise ValueError(f"{spec} does not fit inside a {height}x{width} image")


def extract_tiles(image: torch.Tensor, specs: Sequence[TileSpec]) -> list[torch.Tensor]:
    """Slice tiles out of a ``[B, C, H, W]`` image.

    Returns one ``[B, C, tile_h, tile_w]`` view (not a copy) per spec.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] image, got shape {tuple(image.shape)}")
    _check_specs_in_bounds(specs, image.shape[2], image.shape[3])
    return [
        image[
            :,
            :,
            spec.top : spec.top + spec.height,
            spec.left : spec.left + spec.width,
        ]
        for spec in specs
    ]


def _neighbor_ramps(
    spec: TileSpec, specs: Sequence[TileSpec], axis: Literal["h", "w"]
) -> tuple[int, int]:
    """Feather ramp lengths (low side, high side) for one tile along one axis.

    The ramp on a side equals the deepest penetration of any overlapping
    neighbour from that side (clipped to the tile extent); sides without a
    neighbour — including those flush with the image boundary — get no ramp.
    """
    if axis == "h":
        start, size = spec.top, spec.height

        def other(s: TileSpec) -> tuple[int, int]:
            return s.top, s.height

        def crosses(s: TileSpec) -> bool:
            return s.left < spec.left + spec.width and spec.left < s.left + s.width
    else:
        start, size = spec.left, spec.width

        def other(s: TileSpec) -> tuple[int, int]:
            return s.left, s.width

        def crosses(s: TileSpec) -> bool:
            return s.top < spec.top + spec.height and spec.top < s.top + s.height

    low = max(
        (
            other(s)[0] + other(s)[1] - start
            for s in specs
            if s is not spec and other(s)[0] < start and crosses(s)
        ),
        default=0,
    )
    high = max(
        (
            start + size - other(s)[0]
            for s in specs
            if s is not spec and other(s)[0] > start and crosses(s)
        ),
        default=0,
    )
    return min(max(low, 0), size), min(max(high, 0), size)


def _axis_window(
    size: int, ramp_low: int, ramp_high: int, device: torch.device
) -> torch.Tensor:
    """Strictly positive 1D Hann feather window, sampled at pixel centres."""
    t = torch.arange(size, dtype=torch.float32, device=device) + 0.5
    window = torch.ones(size, dtype=torch.float32, device=device)
    if ramp_low > 0:
        window = window * torch.sin((t / ramp_low).clamp(max=1.0) * (math.pi / 2)) ** 2
    if ramp_high > 0:
        window = (
            window
            * torch.sin(((size - t) / ramp_high).clamp(max=1.0) * (math.pi / 2)) ** 2
        )
    return window


def stitch_tiles(
    tiles: Sequence[torch.Tensor],
    specs: Sequence[TileSpec],
    height: int,
    width: int,
) -> torch.Tensor:
    """Blend ``[B, C, tile_h, tile_w]`` tiles back into a ``[B, C, H, W]`` image.

    Overlap regions are feathered with separable raised-cosine windows (see
    module docstring); the specs must cover every pixel of the output at least
    once (always true for :func:`plan_tiles` output).

    Intended for float images (``[0, 1]`` range per repo convention); integer
    dtypes are cast back by truncation, so blend in float and quantize
    afterwards if exact integer roundtrips matter.
    """
    if len(tiles) != len(specs):
        raise ValueError(f"Got {len(tiles)} tiles but {len(specs)} specs")
    if not tiles:
        raise ValueError("Cannot stitch an empty tile list")
    _check_specs_in_bounds(specs, height, width)
    batch, channels = tiles[0].shape[:2]
    device = tiles[0].device
    canvas = torch.zeros(
        batch, channels, height, width, dtype=torch.float32, device=device
    )
    weight = torch.zeros(height, width, dtype=torch.float32, device=device)
    for tile, spec in zip(tiles, specs, strict=True):
        expected = (batch, channels, spec.height, spec.width)
        if tuple(tile.shape) != expected:
            raise ValueError(
                f"Tile shape {tuple(tile.shape)} does not match {spec} "
                f"(expected {expected})"
            )
        ramp_top, ramp_bottom = _neighbor_ramps(spec, specs, "h")
        ramp_left, ramp_right = _neighbor_ramps(spec, specs, "w")
        window_h = _axis_window(spec.height, ramp_top, ramp_bottom, device)
        window_w = _axis_window(spec.width, ramp_left, ramp_right, device)
        window = einsum(window_h, window_w, "h, w -> h w")
        rows = slice(spec.top, spec.top + spec.height)
        cols = slice(spec.left, spec.left + spec.width)
        canvas[:, :, rows, cols] += tile.float() * window
        weight[rows, cols] += window
    if not torch.all(weight > 0):
        uncovered = int((weight == 0).sum().item())
        raise ValueError(
            f"Tile specs leave {uncovered} of {height * width} pixels uncovered; "
            "stitching requires full coverage (use plan_tiles to generate specs)"
        )
    stitched = canvas / rearrange(weight, "h w -> 1 1 h w")
    return stitched.to(tiles[0].dtype)


if __name__ == "__main__":
    from rich import print

    torch.manual_seed(0)

    # a) Odd-sized image: split -> stitch must reproduce the input exactly,
    #    and the feather windows of a plan grid must sum to exactly 1.
    image = torch.rand(1, 3, 733, 1021)
    specs = plan_tiles(733, 1021, tile_size=512, overlap=96)
    restored = stitch_tiles(extract_tiles(image, specs), specs, 733, 1021)
    max_err = (restored - image).abs().max().item()
    assert torch.allclose(restored, image, atol=1e-5), max_err
    weight = torch.zeros(733, 1021)
    for spec in specs:
        window_h = _axis_window(
            spec.height, *_neighbor_ramps(spec, specs, "h"), weight.device
        )
        window_w = _axis_window(
            spec.width, *_neighbor_ramps(spec, specs, "w"), weight.device
        )
        weight[
            spec.top : spec.top + spec.height, spec.left : spec.left + spec.width
        ] += einsum(window_h, window_w, "h, w -> h w")
    assert torch.allclose(weight, torch.ones_like(weight), atol=1e-5), (
        weight.min().item(),
        weight.max().item(),
    )
    print(
        f"[green]roundtrip ok[/green]: 733x1021, tile 512/overlap 96 -> "
        f"{len(specs)} tiles, max abs err {max_err:.2e}, "
        f"weight sum in [{weight.min().item():.6f}, {weight.max().item():.6f}]"
    )

    # overlap=0 boundary of the validated range: still an exact roundtrip
    # (the evenly-spread layout may keep a positive actual overlap).
    specs = plan_tiles(733, 1021, tile_size=512, overlap=0)
    restored = stitch_tiles(extract_tiles(image, specs), specs, 733, 1021)
    assert torch.allclose(restored, image, atol=1e-5)
    print(f"[green]overlap=0 ok[/green]: {len(specs)} tiles, exact roundtrip")

    # b) 4K plan: full coverage, adjacent overlap >= requested, edges flush.
    plan_h, plan_w, tile_size, overlap = 2160, 3840, 1024, 192
    specs = plan_tiles(plan_h, plan_w, tile_size=tile_size, overlap=overlap)
    coverage = torch.zeros(plan_h, plan_w, dtype=torch.int32)
    for spec in specs:
        coverage[
            spec.top : spec.top + spec.height, spec.left : spec.left + spec.width
        ] += 1
    assert int(coverage.min().item()) >= 1, "4K plan leaves uncovered pixels"
    tops = sorted({spec.top for spec in specs})
    lefts = sorted({spec.left for spec in specs})
    for starts, extent in ((tops, plan_h), (lefts, plan_w)):
        assert starts[0] == 0 and starts[-1] + tile_size == extent
        for prev, nxt in zip(starts, starts[1:], strict=False):
            actual = prev + tile_size - nxt
            assert actual >= overlap, f"adjacent overlap {actual} < {overlap}"
    print(
        f"[green]4K plan ok[/green]: {plan_h}x{plan_w}, tile {tile_size}/overlap "
        f"{overlap} -> {len(tops)}x{len(lefts)} grid, min coverage "
        f"{int(coverage.min().item())}, max coverage {int(coverage.max().item())}"
    )

    # c) Image smaller than tile_size degenerates to a single full-image tile.
    specs = plan_tiles(256, 256, tile_size=512, overlap=96)
    assert specs == [TileSpec(top=0, left=0, height=256, width=256)]
    small = torch.rand(2, 3, 256, 256)
    restored = stitch_tiles(extract_tiles(small, specs), specs, 256, 256)
    assert torch.allclose(restored, small, atol=1e-5)
    print("[green]single-tile fallback ok[/green]: 256x256 with tile 512")
