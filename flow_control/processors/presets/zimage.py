from typing import Literal

from pydantic import BaseModel

from flow_control.utils.resize import ResolutionList

from ..components.encoder import Encoder, Qwen3Encoder
from ..components.prompts import PromptStr
from ..components.vae import VAE, Flux1VAE
from .base import preset_registry


@preset_registry.register("zimage")
class ZImagePreset(BaseModel):
    vae: VAE = Flux1VAE()
    encoder: Encoder = Qwen3Encoder()

    # https://github.com/Tongyi-MAI/Z-Image says the model support arbitrary resolutions
    # between 512x512 and 2048x2048
    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = [
        # 720p
        (720, 720),
        (896, 512),
        (512, 896),
        (832, 544),
        (544, 832),
        (800, 576),
        (576, 800),
        # 1024p
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1152, 864),
        (864, 1152),
        (1248, 832),
        (832, 1248),
        (1280, 720),
        (720, 1280),
        (1344, 576),
        (576, 1344),
        # 1280p
        (1280, 1280),
        (1440, 1120),
        (1120, 1440),
        (1472, 1104),
        (1104, 1472),
        (1536, 1024),
        (1024, 1536),
        (1536, 864),
        (864, 1536),
        (1680, 720),
        (720, 1680),
    ]
    multiple_of: int = 32
    total_pixels: int = 0

    encoder_prompt: PromptStr = ""
    default_negative_prompt: PromptStr = ""
    save_negative: bool = True
