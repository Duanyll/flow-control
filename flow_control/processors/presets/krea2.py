from typing import Literal

import torch
from pydantic import BaseModel

from flow_control.utils.hf_model import HfModelLoader

from ..components.encoder import Encoder, Qwen3VLEncoder
from ..components.prompts import PromptStr
from ..components.vae import VAE, QwenImageVAE
from .base import preset_registry

KREA2_RAW = "krea/Krea-2-Raw"
KREA2_TURBO = "krea/Krea-2-Turbo"


@preset_registry.register("krea2_raw")
class Krea2RawPreset(BaseModel):
    """Krea 2 Raw (undistilled 12.9B single-stream DiT).

    Krea conditions the transformer on a 4D ``prompt_embeds`` (12 stacked Qwen3-VL layers,
    collapsed internally by ``Krea2TextFusion``) plus a ``prompt_embeds_mask`` (padding
    kept). We reuse the ``encoder`` slot for :class:`Qwen3VLEncoder` and override
    :meth:`encode_prompt` to emit both tensors. Raw is a true-CFG model
    (``save_negative=True``), so the negative prompt is encoded through the same override.
    The VAE is the Qwen-Image autoencoder, reused via :class:`QwenImageVAE`.
    """

    vae: VAE = QwenImageVAE(pretrained_model_id=KREA2_RAW, subfolder="vae")
    encoder: Qwen3VLEncoder = Qwen3VLEncoder(
        pretrained_model_id=KREA2_RAW,
        subfolder="text_encoder",
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="Qwen2Tokenizer",
            pretrained_model_id=KREA2_RAW,
            subfolder="tokenizer",
        ),
    )
    pooled_encoder: Encoder | None = None

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "multiple_of"
    multiple_of: int = 32
    total_pixels: int = 1024 * 1024

    encoder_prompt: PromptStr = ""
    default_negative_prompt: PromptStr = ""
    save_negative: bool = True

    def encode_prompt(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, torch.Tensor | None]:
        embeds, mask = self.encoder.encode_seq_mask(prompt)
        return {"prompt_embeds": embeds, "prompt_embeds_mask": mask}


@preset_registry.register("krea2_turbo")
class Krea2TurboPreset(Krea2RawPreset):
    """Krea 2 Turbo (8-step distilled): guidance-free, so no negative branch."""

    vae: VAE = QwenImageVAE(pretrained_model_id=KREA2_TURBO, subfolder="vae")
    encoder: Qwen3VLEncoder = Qwen3VLEncoder(
        pretrained_model_id=KREA2_TURBO,
        subfolder="text_encoder",
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="Qwen2Tokenizer",
            pretrained_model_id=KREA2_TURBO,
            subfolder="tokenizer",
        ),
    )
    save_negative: bool = False
