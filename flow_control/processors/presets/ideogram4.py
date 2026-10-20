from typing import Literal

from pydantic import BaseModel

from ..components.encoder import Encoder, Ideogram4Encoder
from ..components.prompts import PromptStr
from ..components.vae import VAE, Ideogram4VAE
from .base import preset_registry


@preset_registry.register("ideogram4")
class Ideogram4Preset(BaseModel):
    """Ideogram 4 (9.3B single-stream DiT, Qwen3-VL-8B features, FLUX.2 VAE).

    Prompts are the structured JSON captions the model was trained on (the official
    ``docs/prompting.md``); plain text is out of distribution and no magic prompt
    runs here. The unconditional branch is a second transformer run image-only, so
    there is no negative prompt (``save_negative=False``). Any multiple of 16 from
    256 to 2048 works; the native quality setting is 2048², the example uses 1024².
    """

    vae: VAE = Ideogram4VAE()
    encoder: Ideogram4Encoder = Ideogram4Encoder()
    pooled_encoder: Encoder | None = None

    latent_channels: int = 32
    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["multiple_of"] = "multiple_of"
    multiple_of: int = 16
    total_pixels: int = 1024 * 1024

    encoder_prompt: PromptStr = ""
    save_negative: bool = False
