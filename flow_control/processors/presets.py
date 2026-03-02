from typing import Literal

from pydantic import BaseModel

from flow_control.utils.resize import ResolutionList

from .components.encoder import (
    ClipTextEncoder,
    Encoder,
    Qwen3Encoder,
    Qwen25VLEncoder,
    T5TextEncoder,
)
from .components.prompts import PromptStr, parse_prompt
from .components.vae import VAE, Flux1VAE, QwenImageVAE

# ---------------------------------- Flux.1 ---------------------------------- #


class Flux1Preset(BaseModel):
    vae: VAE = Flux1VAE()
    encoder: Encoder = T5TextEncoder()
    pooled_encoder: Encoder | None = ClipTextEncoder()

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = [
        (672, 1568),
        (688, 1504),
        (720, 1456),
        (752, 1392),
        (800, 1328),
        (832, 1248),
        (880, 1184),
        (944, 1104),
        (1024, 1024),
        (1104, 944),
        (1184, 880),
        (1248, 832),
        (1328, 800),
        (1392, 752),
        (1456, 720),
        (1504, 688),
        (1568, 672),
    ]
    multiple_of: int = 32
    total_pixels: int = 1024 * 1024

    encoder_prompt: PromptStr = ""
    save_negative: bool = False


# -------------------------------- Qwen-Image -------------------------------- #


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


class QwenImageEditPreset(QwenImagePreset):
    encoder_prompt: PromptStr = parse_prompt("@qwen_image_edit_encoder")


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


# ------------------------------- Longcat Image ------------------------------ #


class LongcatImagePreset(BaseModel):
    vae: VAE = Flux1VAE()
    encoder: Encoder = Qwen25VLEncoder(
        pretrained_model_id="meituan-longcat/LongCat-Image",
        subfolder="text_encoder",
        image_template="<|vision_start|><|image_pad|><|vision_end|>",
        split_quotation=True,
        drop_suffix_tokens=True,
        tokenizer_max_length=512,
        tokenizer_enforce_pad_token=True,
    )

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = [
        (672, 1568),
        (688, 1504),
        (720, 1456),
        (752, 1392),
        (800, 1328),
        (832, 1248),
        (880, 1184),
        (944, 1104),
        (1024, 1024),
        (1104, 944),
        (1184, 880),
        (1248, 832),
        (1328, 800),
        (1392, 752),
        (1456, 720),
        (1504, 688),
        (1568, 672),
    ]
    multiple_of: int = 32
    total_pixels: int = 1024 * 1024

    encoder_prompt: PromptStr = parse_prompt("@longcat_image_encoder")
    default_negative_prompt: PromptStr = ""
    save_negative: bool = True


class LongcatImageEditPreset(LongcatImagePreset):
    encoder_prompt: PromptStr = parse_prompt("@longcat_image_edit_encoder")


# ---------------------------------- Z-Image --------------------------------- #


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
