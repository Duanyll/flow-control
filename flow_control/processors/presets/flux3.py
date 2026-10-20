import math
from typing import Literal

import torch
from pydantic import BaseModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.resize import resize_to_resolution

from ..components.encoder import Encoder, Flux3Encoder
from ..components.prompts import FLUX3_I2I_SCAFFOLD, PromptStr, parse_prompt
from ..components.vae import VAE, Flux2VAE
from .base import preset_registry

FLUX3_BASE = "black-forest-labs/flux-3-image-klein-base"
FLUX3_DISTILLED = "black-forest-labs/flux-3-image-klein"
FLUX3_REFERENCE_PIXELS = 4 * 1024 * 1024
FLUX3_MULTI_REFERENCE_PIXELS = 1024 * 1024
FLUX3_MAX_SIDE = 4096


def _flux3_encoder(repo: str) -> Flux3Encoder:
    return Flux3Encoder(
        pretrained_model_id=repo,
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="Qwen3VLProcessor",
            pretrained_model_id=repo,
            subfolder="text_encoder",
        ),
    )


@preset_registry.register("flux3_klein_base")
class Flux3KleinBasePreset(BaseModel):
    """FLUX 3 Image Klein base (undistilled): true CFG 3.5 against the empty prompt.

    Text is the :class:`Flux3Encoder` recipe (8 Qwen3-VL layers, padded to 80, no
    mask); the VAE is the FLUX.2 autoencoder shipped with the repo, run in fp32 like
    the official code. Images and references are multiples of 16; a canvas taken
    from the first reference is capped near 4 MP like the official single-reference
    canvas (with several references the official code generates at the first
    reference's <= 1 MP size instead; pass ``image_size`` for that).
    The model was trained on upsampler captions only: give it long, literal
    descriptions, or set ``enable_enhance`` with an ``llm`` serving the official
    upsampler (``black-forest-labs/flux-3-image-klein-pup``); the enhance prompts
    default to its system prompts and edits get its scaffold (see the examples).
    """

    vae: VAE = Flux2VAE(
        pretrained_model_id=FLUX3_BASE, subfolder="vae", dtype=torch.float32
    )
    encoder: Encoder = _flux3_encoder(FLUX3_BASE)
    pooled_encoder: Encoder | None = None

    latent_channels: int = 32

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["multiple_of"] = "multiple_of"
    multiple_of: int = 16
    total_pixels: int = FLUX3_REFERENCE_PIXELS
    no_upscale: bool = True
    max_reference_images: int = 10

    encoder_prompt: PromptStr = ""
    default_negative_prompt: PromptStr = ""
    save_negative: bool = True

    t2i_enhance_prompt: PromptStr = parse_prompt("@flux3_t2i_enhance")
    tie_enhance_prompt: PromptStr = parse_prompt("@flux3_tie_enhance")

    def enhance_input(self, prompt: str) -> str:
        """The upsampler's i2i scaffold around the edit instruction (official ``pup.py``)."""
        return FLUX3_I2I_SCAFFOLD.format(instruction=prompt)

    def resize_reference_images(
        self, reference_images: list[torch.Tensor], image_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        """Official sizing: one reference fits under 4 MP, each of several under 1 MP;
        then a centre crop to the 16-px grid (the official single-reference path
        reflect-pads instead and crops the output back, which is not reproduced)."""
        limit = (
            FLUX3_REFERENCE_PIXELS
            if len(reference_images) == 1
            else FLUX3_MULTI_REFERENCE_PIXELS
        )
        resized = []
        for image in reference_images:
            _, _, h, w = image.shape
            scale = min(1.0, math.sqrt(limit / (h * w)), FLUX3_MAX_SIDE / max(h, w))
            size = (int(h * scale) // 16 * 16, int(w * scale) // 16 * 16)
            resized.append(resize_to_resolution(image, size))
        return resized


@preset_registry.register("flux3_klein_distilled")
class Flux3KleinDistilledPreset(Flux3KleinBasePreset):
    """FLUX 3 Image Klein distilled (16-step flow map): guidance-free, no negative."""

    vae: VAE = Flux2VAE(
        pretrained_model_id=FLUX3_DISTILLED, subfolder="vae", dtype=torch.float32
    )
    encoder: Encoder = _flux3_encoder(FLUX3_DISTILLED)
    save_negative: bool = False
