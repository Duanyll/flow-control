from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from flow_control.utils.hf_model import HfModelLoader

from ..components.encoder import Encoder, Sd3ClipEncoder, T5TextEncoder
from ..components.prompts import PromptStr
from ..components.vae import VAE, Flux1VAE
from .base import preset_registry

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
    :meth:`encode_prompt` to build the fused tensors, and list all three text
    encoders in ``_encoding_components`` (the default loads only the ``encoder`` /
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
