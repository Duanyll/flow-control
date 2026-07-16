# Vendored from https://github.com/HiDream-ai/HiDream-O1-Image
# (commit 2c2d29ff729e48f33e41f49edfdbd81d5ac103b4, MIT license).
# See qwen3_vl_transformers.py / utils.py headers for local changes.
from .qwen3_vl_transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel
from .utils import (
    PREDEFINED_RESOLUTIONS,
    calculate_dimensions,
    get_rope_index_fix_point,
)

__all__ = [
    "PREDEFINED_RESOLUTIONS",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "calculate_dimensions",
    "get_rope_index_fix_point",
]
