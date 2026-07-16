from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.registry import Registry
from flow_control.utils.resize import ResolutionList, resize_to_multiple_of

from .components.encoder import (
    ClipTextEncoder,
    Encoder,
    HiDreamO1Encoder,
    Mistral3Encoder,
    Qwen3Encoder,
    Qwen3VLEncoder,
    Qwen25VLEncoder,
    Sd3ClipEncoder,
    T5TextEncoder,
)
from .components.prompts import PromptStr, parse_prompt
from .components.vae import VAE, Flux1VAE, Flux2VAE, IdentityVAE, QwenImageVAE

FLUX1_RESOLUTIONS = [
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

preset_registry: Registry = Registry("processor_preset")

# ---------------------------------- Flux.1 ---------------------------------- #


@preset_registry.register("flux1")
class Flux1Preset(BaseModel):
    vae: VAE = Flux1VAE()
    encoder: Encoder = T5TextEncoder()
    pooled_encoder: Encoder | None = ClipTextEncoder()

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = FLUX1_RESOLUTIONS
    multiple_of: int = 32
    total_pixels: int = 1024 * 1024
    max_reference_images: int = 4

    encoder_prompt: PromptStr = ""
    save_negative: bool = False


# -------------------------------- Qwen-Image -------------------------------- #


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


# ------------------------------- Longcat Image ------------------------------ #


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


# ---------------------------------- Z-Image --------------------------------- #


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


# ---------------------------------- Flux.2 ---------------------------------- #


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


# ------------------------ Stable Diffusion 3.5 Medium ----------------------- #

SD35_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"


@preset_registry.register("sd35_medium")
class Sd35MediumPreset(BaseModel):
    """Stable Diffusion 3.5 Medium (MMDiT with three text encoders).

    SD3 conditions the transformer on two tensors that line up exactly with the
    ``{prompt_embeds, pooled_prompt_embeds}`` contract:

    - ``prompt_embeds`` = ``cat([pad(cat([clipL_seq, clipG_seq]) -> 4096), t5_seq])``
    - ``pooled_prompt_embeds`` = ``cat([clipL_pooled, clipG_pooled])``

    We reuse the base ``encoder`` slot for T5 (the long-context sequence encoder, as
    FLUX does) and add CLIP-L / CLIP-G as extra fields, then override
    :meth:`encode_prompt` to build the fused tensors and :meth:`load_models` so all
    three text encoders are loaded (the default only loads the ``encoder`` /
    ``pooled_encoder`` slots).
    """

    vae: VAE = Flux1VAE(pretrained_model_id=SD35_MEDIUM, subfolder="vae")
    encoder: Encoder = T5TextEncoder(
        pretrained_model_id=SD35_MEDIUM,
        subfolder="text_encoder_3",
        max_length=256,
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="T5TokenizerFast",
            pretrained_model_id=SD35_MEDIUM,
            subfolder="tokenizer_3",
        ),
    )
    pooled_encoder: Encoder | None = None
    clip_l: Sd3ClipEncoder = Sd3ClipEncoder(
        subfolder="text_encoder",
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="CLIPTokenizer",
            pretrained_model_id=SD35_MEDIUM,
            subfolder="tokenizer",
        ),
    )
    clip_g: Sd3ClipEncoder = Sd3ClipEncoder(
        subfolder="text_encoder_2",
        tokenizer=HfModelLoader(
            library="transformers",
            class_name="CLIPTokenizer",
            pretrained_model_id=SD35_MEDIUM,
            subfolder="tokenizer_2",
        ),
    )

    _encoding_components: list[str] = ["vae", "encoder", "clip_l", "clip_g"]

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["list", "multiple_of"] = "multiple_of"
    multiple_of: int = 64
    total_pixels: int = 1024 * 1024

    encoder_prompt: PromptStr = ""
    default_negative_prompt: PromptStr = ""
    save_negative: bool = True

    def model_post_init(self, context: object, /) -> None:
        # Force the per-instance encode set. Declaring the private attr default is not
        # enough: when the task+preset mixin class is built dynamically, Pydantic's
        # private-attr merge keeps ``BaseProcessor``'s default, so we set it here.
        super().model_post_init(context)
        self._encoding_components = ["vae", "encoder", "clip_l", "clip_g"]

    def encode_prompt(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, torch.Tensor | None]:
        t5_seq = self.encoder.encode(prompt)
        clip_l_seq, clip_l_pooled = self.clip_l.encode_seq_pooled(prompt)
        clip_g_seq, clip_g_pooled = self.clip_g.encode_seq_pooled(prompt)

        clip_seq = torch.cat([clip_l_seq, clip_g_seq], dim=-1)
        clip_seq = F.pad(clip_seq, (0, t5_seq.shape[-1] - clip_seq.shape[-1]))
        prompt_embeds = torch.cat([clip_seq, t5_seq], dim=-2)
        pooled_prompt_embeds = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }


# ----------------------------------- Krea 2 ---------------------------------- #

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


# --------------------------------- HiDream-O1 -------------------------------- #

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
    :meth:`get_latent_length` is overridden (the task implementations read
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

    def get_latent_length(self, batch: Mapping[str, Any]) -> int:
        h, w = batch["image_size"]
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        length = (h * w) // ratio + batch["input_ids"].shape[1]
        for rh, rw in batch.get("reference_sizes") or []:
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
