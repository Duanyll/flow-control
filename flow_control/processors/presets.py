from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.registry import Registry
from flow_control.utils.resize import ResolutionList

from .components.encoder import (
    ClipTextEncoder,
    Encoder,
    Mistral3Encoder,
    Qwen3Encoder,
    Qwen25VLEncoder,
    Sd3ClipEncoder,
    T5TextEncoder,
)
from .components.prompts import PromptStr, parse_prompt
from .components.vae import VAE, Flux1VAE, Flux2VAE, QwenImageVAE

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
