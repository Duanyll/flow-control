from typing import Literal

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from pydantic import BaseModel

from flow_control.utils.resize import ResolutionList
from flow_control.utils.tensor import tensor_to_pil

from ..components.encoder import Encoder, QwenImage21Encoder
from ..components.prompts import PromptStr, parse_prompt
from ..components.vae import VAE, QwenImage21VAE
from .base import preset_registry


@preset_registry.register("qwen21")
class QwenImage21Preset(BaseModel):
    vae: VAE = QwenImage21VAE()
    encoder: QwenImage21Encoder = QwenImage21Encoder()
    pooled_encoder: Encoder | None = None
    patch_size: int = 1
    vae_scale_factor: int = 16
    latent_channels: int = 64
    default_resolution: tuple[int, int] = (2048, 2048)
    resize_mode: Literal["list"] = "list"
    preferred_resolutions: ResolutionList = [
        (2048, 2048),
        (1792, 2400),
        (2400, 1792),
        (1696, 2528),
        (2528, 1696),
        (1536, 2752),
        (2752, 1536),
    ]
    multiple_of: int = 32
    total_pixels: int = 2048 * 2048
    max_reference_images: int = 10
    encoder_prompt: PromptStr = ""
    t2i_enhance_prompt: PromptStr = parse_prompt("@qwen21_t2i_enhance")
    tie_enhance_prompt: PromptStr = parse_prompt("@qwen21_tie_enhance")
    enhance_json_field: str = "rewritten_prompt"
    default_negative_prompt: str = " "
    save_negative: bool = False
    negative_with_images: bool = True

    def encode_prompt(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, torch.Tensor | None]:
        embeds, mask, image_mask = self.encoder.encode_condition(prompt, images)
        return {
            "prompt_embeds": embeds,
            "prompt_embeds_mask": mask,
            "image_pad_mask": image_mask,
            "pooled_prompt_embeds": None,
        }

    def resize_reference_images(
        self, reference_images: list[torch.Tensor], image_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        # Match the official PIL resize, including RGBA's alpha handling.
        # Both VLM and VAE must receive the same pixels and 32-aligned layout.
        processor = VaeImageProcessor(vae_scale_factor=32, do_normalize=False)
        resized = []
        for image in reference_images:
            pil = tensor_to_pil(image).convert("RGBA")
            width, height, _ = calculate_dimensions(
                self.total_pixels, pil.width / pil.height
            )
            resized.append(
                processor.preprocess(pil, height=height, width=width).to(image.device)
            )
        return resized
