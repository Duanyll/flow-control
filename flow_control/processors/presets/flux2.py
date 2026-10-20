from typing import Literal

import torch
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from pydantic import BaseModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.resize import ResolutionList
from flow_control.utils.tensor import tensor_to_pil

from ..components.encoder import Encoder, Mistral3Encoder, Qwen3Encoder
from ..components.prompts import PromptStr, parse_prompt
from ..components.vae import VAE, Flux2VAE
from .base import FLUX1_RESOLUTIONS, preset_registry


@preset_registry.register("flux2")
class Flux2Preset(BaseModel):
    vae: VAE = Flux2VAE()
    encoder: Encoder = Mistral3Encoder()

    latent_channels: int = 32

    # Flux.2 can support arbitrary resolutions between 0.1MP and 2MP
    # https://www.reddit.com/r/StableDiffusion/comments/1enxdga/flux_recommended_resolutions_from_01_to_20/
    # https://docs.google.com/spreadsheets/d/1p913YOU9A6rC0nasQPvKWsNDrE-OOUHU4-AZI8Eqois/edit?usp=sharing
    default_resolution: tuple[int, int] = (1024, 1024)
    preferred_resolutions: ResolutionList = FLUX1_RESOLUTIONS
    resize_mode: Literal["multiple_of"] = "multiple_of"
    multiple_of: int = 32
    total_pixels: int = 0
    max_reference_images: int = 10

    encoder_prompt: PromptStr = parse_prompt("@flux2_encoder")
    default_negative_prompt: PromptStr = ""
    save_negative: bool = False

    t2i_enhance_prompt: PromptStr = parse_prompt("@flux2_t2i_enhance")
    tie_enhance_prompt: PromptStr = parse_prompt("@flux2_tie_enhance")


@preset_registry.register("flux2_klein_9b")
class Flux2Klein9BPreset(Flux2Preset):
    encoder: Encoder = Qwen3Encoder(
        pretrained_model_id="black-forest-labs/FLUX.2-klein-9B",
        subfolder="text_encoder",
        hidden_state_layers=[9, 18, 27],
        enable_thinking=False,
        keep_padding_tokens=True,
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="Qwen2TokenizerFast",
            pretrained_model_id="black-forest-labs/FLUX.2-klein-9B",
            subfolder="tokenizer",
        ),
    )

    encoder_prompt: PromptStr = ""


@preset_registry.register("flux2_klein_9b_kv")
class Flux2Klein9BKVPreset(Flux2Klein9BPreset):
    vae: VAE = Flux2VAE(pretrained_model_id="black-forest-labs/FLUX.2-klein-9b-kv")
    encoder: Encoder = Qwen3Encoder(
        pretrained_model_id="black-forest-labs/FLUX.2-klein-9b-kv",
        subfolder="text_encoder",
        hidden_state_layers=[9, 18, 27],
        enable_thinking=False,
        keep_padding_tokens=True,
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="Qwen2TokenizerFast",
            pretrained_model_id="black-forest-labs/FLUX.2-klein-9b-kv",
            subfolder="tokenizer",
        ),
    )
    reference_image_resize_mode: Literal["multiple_of"] = "multiple_of"

    def resize_reference_images(
        self, reference_images: list[torch.Tensor], image_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        processor = Flux2ImageProcessor(do_normalize=False)
        resized = []
        for image in reference_images:
            pil = tensor_to_pil(image)
            processor.check_image_input(pil)
            if pil.width * pil.height > 1024**2:
                pil = processor._resize_to_target_area(pil, 1024**2)
            resized.append(
                processor.preprocess(
                    pil,
                    height=pil.height // 16 * 16,
                    width=pil.width // 16 * 16,
                    resize_mode="crop",
                ).to(image.device)
            )
        return resized


@preset_registry.register("flux2_klein_4b")
class Flux2Klein4BPreset(Flux2Preset):
    encoder: Encoder = Qwen3Encoder(
        pretrained_model_id="black-forest-labs/FLUX.2-klein-4B",
        subfolder="text_encoder",
        hidden_state_layers=[9, 18, 27],
        enable_thinking=False,
        keep_padding_tokens=True,
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="Qwen2TokenizerFast",
            pretrained_model_id="black-forest-labs/FLUX.2-klein-4B",
            subfolder="tokenizer",
        ),
    )

    encoder_prompt: PromptStr = ""
