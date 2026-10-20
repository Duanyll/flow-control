from collections.abc import Mapping
from typing import Any, Literal

import torch
from pydantic import BaseModel

from flow_control.utils.resize import ResolutionList, resize_to_multiple_of

from ..components.encoder import Encoder, HiDreamO1Encoder
from ..components.prompts import PromptStr
from ..components.vae import VAE, IdentityVAE
from .base import preset_registry

HIDREAM_O1_FULL = "HiDream-ai/HiDream-O1-Image"
HIDREAM_O1_DEV = "HiDream-ai/HiDream-O1-Image-Dev"

# Official PREDEFINED_RESOLUTIONS transposed from (W, H) to the framework's (H, W).
HIDREAM_O1_RESOLUTIONS = [
    (2048, 2048),
    (1728, 2304),
    (2304, 1728),
    (1440, 2560),
    (2560, 1440),
    (1664, 2496),
    (2496, 1664),
    (1312, 3104),
    (3104, 1312),
    (1792, 2304),
    (2304, 1792),
]


@preset_registry.register("hidream_o1_full")
class HiDreamO1FullPreset(BaseModel):
    """HiDream-O1 (pixel-space unified transformer, no VAE, no text encoder).

    Latents are 1/8-scaled pixels (:class:`IdentityVAE`), packed 32x32, so the
    processor geometry is ``vae_scale_factor=1, patch_size=32,
    latent_channels=3``. Text conditioning is token ids consumed by the same
    trainable transformer, so :meth:`encode_prompt` emits ``input_ids`` (plus
    ``pixel_values``/``image_grid_thw`` SigLIP-condition thumbnails on the
    editing path) instead of embeddings — which is also why
    :meth:`get_cost` is overridden (the task implementations read
    ``prompt_embeds``).

    The full checkpoint is undistilled: true CFG 5.0 with ``" "`` as the
    unconditional prompt, and on editing tasks the unconditional pass keeps the
    reference images (``negative_with_images``).
    """

    vae: VAE = IdentityVAE()
    encoder: HiDreamO1Encoder = HiDreamO1Encoder()
    pooled_encoder: Encoder | None = None

    vae_scale_factor: int = 1
    patch_size: int = 32
    latent_channels: int = 3
    initial_noise_scale: float = 0.9375
    """The official pipeline initializes sampling noise at 7.5/8 of the model's
    pixel-space noise scale (``--noise_scale_start`` default 7.5)."""

    default_resolution: tuple[int, int] = (2048, 2048)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = HIDREAM_O1_RESOLUTIONS
    multiple_of: int = 32
    total_pixels: int = 2048 * 2048

    encoder_prompt: PromptStr = ""
    default_negative_prompt: PromptStr = " "
    save_negative: bool = True
    negative_with_images: bool = True

    def encode_prompt(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if images:
            return self.encoder.encode_ids_with_images(prompt, images, system_prompt)
        return {"input_ids": self.encoder.encode_ids(prompt, system_prompt)}

    def get_cost(self, row: Mapping[str, Any]) -> int:
        h, w = row["image_size"]
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        length = (h * w) // ratio + row["input_ids"].shape[1]
        for rh, rw in row.get("reference_sizes") or []:
            length += (rh * rw) // ratio
        return length

    def resize_reference_images(
        self, reference_images: list[torch.Tensor], image_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        """Official K-count area heuristic for the in-sequence reference patches
        (the SigLIP thumbnails are sized separately inside the encoder)."""
        k = len(reference_images)
        if k == 0:
            return reference_images
        max_side = max(image_size)
        if k == 1:
            max_size = max_side
        elif k == 2:
            max_size = max_side * 48 // 64
        elif k <= 4:
            max_size = max_side // 2
        elif k <= 8:
            max_size = max_side * 24 // 64
        else:
            max_size = max_side // 4
        return [
            resize_to_multiple_of(img, multiple=self.patch_size, pixels=max_size**2)
            for img in reference_images
        ]


@preset_registry.register("hidream_o1_dev")
class HiDreamO1DevPreset(HiDreamO1FullPreset):
    """HiDream-O1-Dev (28-step distilled): guidance-free, flash re-noise sampler."""

    encoder: HiDreamO1Encoder = HiDreamO1Encoder(pretrained_model_id=HIDREAM_O1_DEV)
    save_negative: bool = False
