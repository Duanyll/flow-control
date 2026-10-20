from typing import Literal

from pydantic import BaseModel

from flow_control.utils.resize import ResolutionList

from ..components.encoder import ClipTextEncoder, Encoder, T5TextEncoder
from ..components.prompts import PromptStr
from ..components.vae import VAE, Flux1VAE
from .base import FLUX1_RESOLUTIONS, preset_registry


@preset_registry.register("flux1")
class Flux1Preset(BaseModel):
    vae: VAE = Flux1VAE()
    encoder: Encoder = T5TextEncoder()
    pooled_encoder: Encoder | None = ClipTextEncoder()

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = FLUX1_RESOLUTIONS
    multiple_of: int = 32
    total_pixels: int = 1024 * 1024
    max_reference_images: int = 4

    encoder_prompt: PromptStr = ""
    save_negative: bool = False
