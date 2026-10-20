from typing import Literal

from pydantic import BaseModel

from ..components.encoder import Cosmos3Encoder, Encoder
from ..components.prompts import PromptStr
from ..components.vae import VAE, Cosmos3VAE
from .base import preset_registry


@preset_registry.register("cosmos3_nano")
class Cosmos3NanoPreset(BaseModel):
    """Cosmos3-Nano (16B) text-to-image.

    ``prompt_embeds`` holds token IDs rather than embeddings -- Cosmos3 has no
    separate text encoder (see :class:`Cosmos3Encoder`). Prompts are meant to be
    JSON-upsampled captions; short benchmark prompts are out of distribution.
    """

    vae: VAE = Cosmos3VAE()
    encoder: Encoder = Cosmos3Encoder()
    pooled_encoder: Encoder | None = None

    patch_size: int = 2
    vae_scale_factor: int = 16
    latent_channels: int = 48
    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["multiple_of"] = "multiple_of"
    # 16x VAE times a 2x2 latent patch: a multiple of 32 keeps the patch grid
    # exact, so nothing has to be zero-padded the way the upstream pipeline does.
    multiple_of: int = 32
    total_pixels: int = 1024 * 1024

    encoder_prompt: PromptStr = ""
    """Empty falls back to the checkpoint's own image-generation system prompt."""
    default_negative_prompt: str = ""
    save_negative: bool = True
    """Cosmos3 runs true CFG (4.0 for Text2Image, 6.0 elsewhere)."""


@preset_registry.register("cosmos3_edge")
class Cosmos3EdgePreset(Cosmos3NanoPreset):
    """Cosmos3-Edge (4B). Only 256p and 480p; no system prompt by default."""

    vae: VAE = Cosmos3VAE(pretrained_model_id="nvidia/Cosmos3-Edge")
    encoder: Encoder = Cosmos3Encoder(
        pretrained_model_id="nvidia/Cosmos3-Edge", use_system_prompt=False
    )
    default_resolution: tuple[int, int] = (480, 832)
    total_pixels: int = 832 * 480


@preset_registry.register("cosmos3_super_t2i")
class Cosmos3SuperText2ImagePreset(Cosmos3NanoPreset):
    """Cosmos3-Super-Text2Image (64B), the text-to-image post-trained expert."""

    vae: VAE = Cosmos3VAE(pretrained_model_id="nvidia/Cosmos3-Super-Text2Image")
    encoder: Encoder = Cosmos3Encoder(
        pretrained_model_id="nvidia/Cosmos3-Super-Text2Image"
    )
