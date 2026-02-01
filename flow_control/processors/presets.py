from typing import Literal

from flow_control.utils.resize import ResolutionList

from .base import BaseProcessor
from .components.encoder import ClipTextEncoder, Encoder, Qwen25VLEncoder, T5TextEncoder
from .components.prompts import PromptStr, parse_prompt
from .components.vae import VAE, Flux1VAE, QwenImageVAE

# ---------------------------------- Flux.1 ---------------------------------- #


class Flux1Processor(BaseProcessor):
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


class QwenImageProcessor(Flux1Processor):
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


class QwenImageEditProcessor(QwenImageProcessor):
    encoder_prompt: PromptStr = parse_prompt("@qwen_image_edit_encoder")


class QwenImageLayeredProcessor(QwenImageProcessor):
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


class LongcatImageProcessor(BaseProcessor):
    vae: VAE = Flux1VAE()
    encoder: Encoder = Qwen25VLEncoder(
        image_template="<|vision_start|><|image_pad|><|vision_end|>",
        split_quotation=True,
        drop_suffix_tokens=True,
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
    save_negative: bool = True


class LongcatImageEditProcessor(LongcatImageProcessor):
    encoder_prompt: PromptStr = parse_prompt("@longcat_image_edit_encoder")
