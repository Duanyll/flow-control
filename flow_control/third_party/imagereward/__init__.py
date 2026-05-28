# Vendored from THUDM/ImageReward (Apache-2.0).
# Source: https://github.com/THUDM/ImageReward
#
# Minimal inference-only subset for reward scoring. Training-time deps
# (fairscale, openai/CLIP) have been stripped.

from .imagereward import ImageReward

__all__ = ["ImageReward"]
