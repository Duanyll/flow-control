from typing import Literal

from pydantic import BaseModel

from flow_control.utils.resize import ResolutionList

from ..components.encoder import Encoder, Qwen25VLEncoder
from ..components.prompts import PromptStr, parse_prompt
from ..components.vae import VAE, Flux1VAE
from .base import FLUX1_RESOLUTIONS, preset_registry


@preset_registry.register("longcat_image")
class LongcatImagePreset(BaseModel):
    vae: VAE = Flux1VAE()
    encoder: Encoder = Qwen25VLEncoder(
        pretrained_model_id="meituan-longcat/LongCat-Image",
        subfolder="text_encoder",
        image_template="<|vision_start|><|image_pad|><|vision_end|>",
        split_quotation=True,
        drop_suffix_tokens=True,
        tokenizer_max_length=512,
        resize_mode="scale",
        keep_padding_tokens=True,
    )

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = FLUX1_RESOLUTIONS
    multiple_of: int = 32
    total_pixels: int = 1024 * 1024

    encoder_prompt: PromptStr = parse_prompt("@longcat_image_encoder")
    default_negative_prompt: PromptStr = ""
    save_negative: bool = True

    t2i_enhance_prompt: PromptStr = parse_prompt("@longcat_t2i_enhance_en")


@preset_registry.register("longcat_image_edit")
class LongcatImageEditPreset(LongcatImagePreset):
    encoder_prompt: PromptStr = parse_prompt("@longcat_image_edit_encoder")
    max_reference_images: int = 1
    negative_with_images: bool = True
    """The official LongCat-Image-Edit pipeline encodes the negative prompt together
    with the reference image."""
