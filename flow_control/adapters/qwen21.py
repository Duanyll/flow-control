import math
from typing import Literal, NotRequired

import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21Rope,
    QwenImage21TemporalTimesteps,
)
from peft import LoraConfig

from flow_control.adapters.base import BaseModelAdapter, Batch, adapter_registry
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.model_cache import cache_enabled


class TemporalTimesteps(QwenImage21TemporalTimesteps):
    """Compute tiny fp32 constants on demand, including after meta/DCP loading."""

    def __init__(self, timestep_dim: int, time_factor: float):
        torch.nn.Module.__init__(self)
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.timestep_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        args = self.time_factor * timestep.float()[:, None] * freqs[None]
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class QwenImage21Batch(Batch):
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    image_pad_mask: torch.Tensor
    reference_latents: NotRequired[list[torch.Tensor]]
    reference_sizes: NotRequired[list[tuple[int, int]]]
    _kv_cache: NotRequired[QwenImage21KVCache]


@adapter_registry.register("qwen21_base")
class QwenImage21Adapter(
    BaseModelAdapter[QwenImage21Transformer2DModel, QwenImage21Batch]
):
    """Qwen-Image 2.1; each executor sample owns its text/reference prefix KV."""

    arch: Literal["qwen21"] = "qwen21"
    type: Literal["base"] = "base"
    patch_size: int = 1
    vae_scale_factor: int = 16
    latent_channels: int = 64

    hf_model: HfModelLoader[QwenImage21Transformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="QwenImage21Transformer2DModel",
        pretrained_model_id="Qwen/Qwen-Image-2.1",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )
    peft_lora_config: LoraConfig = LoraConfig(
        target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
    )

    def _install_modules(self) -> None:
        # RoPE's ordinary tensor list survives to_empty, but must be created
        # outside the trainer's meta context. Keep it complex64: registering
        # these tables as buffers would let model.to(bf16) discard their phase.
        with torch.device("cpu"):
            rope = self.transformer.pos_embed
            self.transformer.pos_embed = QwenImage21Rope(rope.theta, rope.axes_dim)
        time_proj = self.transformer.time_text_embed.time_proj
        self.transformer.time_text_embed.time_proj = TemporalTimesteps(
            time_proj.timestep_dim, time_proj.time_factor
        )

    def _predict_velocity(
        self, batch: QwenImage21Batch, timestep: torch.Tensor
    ) -> torch.Tensor:
        latents = batch["noisy_latents"]
        b, n, _ = latents.shape
        hidden_states = torch.cat([*batch.get("reference_latents", []), latents], dim=1)
        shapes = [
            (1, h // self.vae_scale_factor, w // self.vae_scale_factor)
            for h, w in [*batch.get("reference_sizes", []), batch["image_size"]]
        ]
        # One VLM image slot represents a 2x2 group of VAE tokens.
        img_mask = torch.cat(
            [
                batch["image_pad_mask"],
                torch.ones(b, n // 4, dtype=torch.bool, device=latents.device),
            ],
            dim=1,
        )
        kv_cache, mode = None, None
        if (
            self.use_cache
            and cache_enabled.get()
            and not torch.is_grad_enabled()
            and self.transformer.config["causal_condition"]
        ):
            kv_cache = batch.get("_kv_cache")
            mode = "cached" if kv_cache is not None else "extract"
            if kv_cache is None:
                kv_cache = QwenImage21KVCache(len(self.transformer.transformer_blocks))
                batch["_kv_cache"] = kv_cache

        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=batch["prompt_embeds"],
            encoder_hidden_states_mask=batch["prompt_embeds_mask"],
            timestep=timestep,
            img_shapes=[shapes] * b,
            img_mask=img_mask,
            kv_cache=kv_cache,
            kv_cache_mode=mode,
            return_dict=False,
        )[0][:, -n:]
