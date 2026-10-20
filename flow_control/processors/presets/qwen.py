from typing import Literal

from pydantic import BaseModel

from flow_control.utils.resize import ResolutionList

from ..components.encoder import Encoder, Qwen25VLEncoder
from ..components.prompts import PromptStr, parse_prompt
from ..components.vae import VAE, QwenImageVAE
from .base import preset_registry


@preset_registry.register("qwen_image")
class QwenImagePreset(BaseModel):
    vae: VAE = QwenImageVAE()
    encoder: Encoder = Qwen25VLEncoder()

    default_resolution: tuple[int, int] = (1328, 1328)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = [
        (1328, 1328),
        (1664, 928),
        (928, 1664),
        (1472, 1104),
        (1104, 1472),
        (1584, 1056),
        (1056, 1584),
    ]
    multiple_of: int = 32
    total_pixels: int = 1328 * 1328

    encoder_prompt: PromptStr = parse_prompt("@qwen_image_encoder")
    save_negative: bool = True


@preset_registry.register("qwen_image_edit")
class QwenImageEditPreset(QwenImagePreset):
    encoder: Encoder = Qwen25VLEncoder(tokenizer_max_length=0)
    encoder_prompt: PromptStr = parse_prompt("@qwen_image_edit_encoder")
    max_reference_images: int = 3
    negative_with_images: bool = True
    """The official Edit and Edit-Plus pipelines encode the negative prompt
    together with the reference images."""


@preset_registry.register("qwen_image_layered")
class QwenImageLayeredPreset(QwenImagePreset):
    vae: VAE = QwenImageVAE(
        pretrained_model_id="Qwen/Qwen-Image-Layered",
        subfolder="vae",
    )

    default_resolution: tuple[int, int] = (640, 640)
    resize_mode: Literal["multiple_of", "list"] = "multiple_of"
    multiple_of: int = 32
    total_pixels: int = 640 * 640
    default_num_layers: int = 4
