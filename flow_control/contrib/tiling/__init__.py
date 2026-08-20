"""Overlapping-tile split / feathered-stitch utilities for large images.

Pure tensor functions (no registry side effects — unlike most ``contrib``
packages there is nothing to self-register); import and call directly::

    from flow_control.contrib.tiling import plan_tiles, extract_tiles, stitch_tiles

    specs = plan_tiles(height, width, tile_size=1024, overlap=192)
    tiles = extract_tiles(image, specs)          # [B, C, H, W] -> list of tiles
    tiles = [enhance(tile) for tile in tiles]    # any per-tile processing
    result = stitch_tiles(tiles, specs, height, width)

See :mod:`.tiling` for the layout and raised-cosine feathering details.
"""

from .tiling import TileSpec, extract_tiles, plan_tiles, stitch_tiles

__all__ = ["TileSpec", "extract_tiles", "plan_tiles", "stitch_tiles"]
