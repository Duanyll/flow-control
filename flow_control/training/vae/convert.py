"""Convert a 3-channel AutoencoderKL (or similar) to 4-channel RGBA.

Ported from AlphaVAE ``models/convert.py``.  Works for any diffusers VAE that
has ``encoder.conv_in`` (3 → N) and ``decoder.conv_out`` (N → 3) layers,
including ``AutoencoderKL`` and ``AutoencoderKLFlux2``.
"""

from typing import Any

import torch
import torch.nn as nn
from diffusers import ModelMixin

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


def convert_to_rgba[T: ModelMixin](model: T) -> T:
    """Expand encoder input from 3ch to 4ch and decoder output from 3ch to 4ch.

    - Encoder ``conv_in``: the extra input channel is zero-initialised.
    - Decoder ``conv_out``: the extra output channel is zero-initialised with
      bias = 1 (opaque default).
    - The diffusers config dict is updated (``in_channels=4``,
      ``out_channels=4``) so that ``save_pretrained`` / ``from_pretrained``
      remain consistent.
    """
    m: Any = model  # escape hatch for dynamic attribute access

    # --- Encoder conv_in: 3 -> 4 input channels ---
    conv_in: nn.Conv2d = m.encoder.conv_in
    if conv_in.in_channels == 4:
        logger.info(
            "Encoder conv_in already has 4 input channels, skipping conversion."
        )
    else:
        conv_in_new = nn.Conv2d(
            4,
            conv_in.out_channels,
            kernel_size=conv_in.kernel_size,  # type: ignore[arg-type]
            stride=conv_in.stride,  # type: ignore[arg-type]
            padding=conv_in.padding,  # type: ignore[arg-type]
        )
        with torch.no_grad():
            conv_in_new.weight[:, :3] = conv_in.weight
            conv_in_new.weight[:, 3:] = 0
            assert conv_in.bias is not None
            assert conv_in_new.bias is not None
            conv_in_new.bias.copy_(conv_in.bias)
        m.encoder.conv_in = conv_in_new
        logger.info("Converted encoder conv_in from 3ch to 4ch input.")

    # --- Decoder conv_out: 3 -> 4 output channels ---
    conv_out: nn.Conv2d = m.decoder.conv_out
    if conv_out.out_channels == 4:
        logger.info(
            "Decoder conv_out already has 4 output channels, skipping conversion."
        )
    else:
        conv_out_new = nn.Conv2d(
            conv_out.in_channels,
            4,
            kernel_size=conv_out.kernel_size,  # type: ignore[arg-type]
            stride=conv_out.stride,  # type: ignore[arg-type]
            padding=conv_out.padding,  # type: ignore[arg-type]
        )
        with torch.no_grad():
            assert conv_out.bias is not None
            assert conv_out_new.bias is not None
            conv_out_new.weight[:3] = conv_out.weight
            conv_out_new.weight[3:] = 0
            conv_out_new.bias[:3] = conv_out.bias
            conv_out_new.bias[3] = 1  # opaque default
        m.decoder.conv_out = conv_out_new
        logger.info("Converted decoder conv_out from 3ch to 4ch output.")

    # --- Update diffusers internal config ---
    if hasattr(m, "_internal_dict"):
        config = dict(m._internal_dict)
        config["in_channels"] = 4
        config["out_channels"] = 4
        m._internal_dict = config
    if hasattr(m, "config") and isinstance(m.config, dict):
        m.config["in_channels"] = 4
        m.config["out_channels"] = 4

    return model
