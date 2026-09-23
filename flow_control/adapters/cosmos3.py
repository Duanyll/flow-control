"""Cosmos3 omni transformer as a flow-matching image generator.

Cosmos3 is a Mixture-of-Transformers: every decoder layer carries two equally
sized experts, an autoregressive *understanding* (und) tower for text and a
diffusion *generation* (gen) tower for image/video/audio/action latents. There
is no separate text encoder -- raw token IDs enter the transformer and the gen
stream cross-attends to the und stream's keys and values at every layer.

The und stream never attends to gen tokens and carries no timestep embedding,
so its per-layer K/V is identical at every denoising step. ``Cosmos3KVAdapter``
extracts that K/V on the first step and replays it afterwards, which skips the
whole und tower. With Cosmos3's JSON-upsampled prompts (1.7k-2.9k tokens
against ~1k image tokens) that is the majority of the per-step work.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, cast

import torch
import torch.nn as nn
from diffusers import Cosmos3OmniTransformer
from diffusers.models.transformers.transformer_cosmos3 import (
    Cosmos3AttnProcessor,
    Cosmos3PackedMoTAttention,
    Cosmos3VLTextRotaryEmbedding,
    _rotate_half,
)
from einops import rearrange, repeat
from peft import LoraConfig
from pydantic import PrivateAttr

from flow_control.adapters.base import BaseModelAdapter, Batch, adapter_registry
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.model_cache import cache_enabled


class Cosmos3Batch(Batch):
    prompt_embeds: torch.Tensor
    """`[1, L]` **Token IDs**, not embeddings.

    Cosmos3 has no text encoder: the preset can only tokenize, and the und
    tower inside the transformer does the encoding. The field keeps the
    conventional name so processors, cost estimation and bucketing work
    unchanged.
    """
    _pack: NotRequired["Cosmos3Pack"]


class Cosmos3KVBatch(Cosmos3Batch):
    _kv_cache: NotRequired["Cosmos3UndKVCache"]


@dataclass
class Cosmos3UndKVCache:
    """Per-layer ``(k_und_for_gen, v_und)`` after qk-norm and rotary embedding."""

    entries: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return bool(self.entries)


@dataclass
class Cosmos3KVState:
    """Shared handle the per-layer processors read during one transformer call."""

    mode: Literal["extract", "cached"] | None = None
    cache: Cosmos3UndKVCache | None = None


class Cosmos3KVAttnProcessor(Cosmos3AttnProcessor):
    """Upstream dual-pathway attention with und K/V extraction and replay.

    The body mirrors ``Cosmos3AttnProcessor.__call__`` step for step so that
    upstream changes stay easy to diff; the only additions are the cache
    branches around the und pathway.
    """

    def __init__(self, layer_index: int, state: Cosmos3KVState):
        self.layer_index = layer_index
        self.state = state

    def __call__(
        self,
        attn: Cosmos3PackedMoTAttention,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        state = self.state
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb

        q_gen = attn.add_q_proj(gen_seq).view(
            -1, attn.num_attention_heads, attn.head_dim
        )
        k_gen = attn.add_k_proj(gen_seq).view(
            -1, attn.num_key_value_heads, attn.head_dim
        )
        v_gen = attn.add_v_proj(gen_seq).view(
            -1, attn.num_key_value_heads, attn.head_dim
        )
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        if state.mode == "cached":
            assert state.cache is not None  # set together with mode
            k_und_for_gen, v_und = state.cache.entries[self.layer_index]
            # The und stream is skipped entirely: `und_seq` is empty, so the
            # layer's residual adds and MLP are no-ops on zero rows.
            und_out = und_seq
        else:
            q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
            k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
            v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
            q_und = attn.norm_q(q_und)
            k_und = attn.norm_k(k_und)
            k_und_for_gen = (
                attn.k_norm_und_for_gen(k_und)
                if attn.k_norm_und_for_gen is not None
                else k_und
            )
            cos_und = cos_und.unsqueeze(1)
            sin_und = sin_und.unsqueeze(1)
            q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
            k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
            k_und_for_gen = (
                k_und_for_gen * cos_und + _rotate_half(k_und_for_gen) * sin_und
            )

            causal_out = dispatch_attention_fn(
                q_und.unsqueeze(0),
                k_und.unsqueeze(0),
                v_und.unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            und_out = attn.to_out(causal_out.squeeze(0).flatten(-2, -1))

            if state.mode == "extract":
                assert state.cache is not None
                state.cache.entries[self.layer_index] = (k_und_for_gen, v_und)

        all_k = torch.cat([k_und_for_gen, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = dispatch_attention_fn(
            q_gen.unsqueeze(0),
            all_k.unsqueeze(0),
            all_v.unsqueeze(0),
            is_causal=False,
            enable_gqa=True,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        gen_out = attn.to_add_out(full_out.squeeze(0).flatten(-2, -1))
        return und_out, gen_out


class Cosmos3RotaryEmbedding(Cosmos3VLTextRotaryEmbedding):
    """Derive the rotary table on every call instead of holding a buffer.

    Upstream registers ``inv_freq`` as a *non-persistent* buffer, so it is
    absent from the DCP seed checkpoint and ``to_empty`` leaves it pointing at
    uninitialized device memory: every rotary embedding comes back garbage and
    the sampler produces flat noise. The table is ``head_dim / 2`` floats fixed
    by the config, so deriving it per call is free and survives meta init,
    ``to_empty`` and DCP load alike -- and, unlike making the buffer persistent,
    it leaves the state dict (and therefore every existing seed and export)
    untouched. ``forward`` already moves it to the compute device.
    """

    def __init__(
        self, head_dim: int, rope_theta: float, rope_axes_dim: Sequence[int]
    ) -> None:
        nn.Module.__init__(self)  # The parent would register the buffer back.
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_axes_dim = rope_axes_dim

    @property
    def inv_freq(self) -> torch.Tensor:
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        return 1.0 / (self.rope_theta ** (exponent / self.head_dim))


@dataclass
class Cosmos3Pack:
    """Everything about one row's joint sequence that does not vary with ``t``.

    ``full`` drives the first (or uncached) step and covers ``[text | vision]``;
    ``vision_only`` drives cached steps, where the text prefix is replaced by
    the cached K/V and the sequence starts at the first vision token.
    """

    input_ids: torch.Tensor
    text_indexes: torch.Tensor
    position_ids: torch.Tensor
    und_len: int
    sequence_length: int
    token_shape: tuple[int, int, int]
    vision_sequence_indexes: torch.Tensor
    vision_mse_loss_indexes: torch.Tensor
    noisy_frame_indexes: torch.Tensor
    vision_position_ids: torch.Tensor

    def transformer_kwargs(self, cached: bool) -> dict[str, object]:
        if not cached:
            return {
                "input_ids": self.input_ids,
                "text_indexes": self.text_indexes,
                "position_ids": self.position_ids,
                "und_len": self.und_len,
                "sequence_length": self.sequence_length,
                "vision_sequence_indexes": self.vision_sequence_indexes,
                "vision_mse_loss_indexes": self.vision_mse_loss_indexes,
            }
        empty = self.input_ids[:0]
        num_vision = self.sequence_length - self.und_len
        return {
            "input_ids": empty,
            "text_indexes": self.text_indexes[:0],
            "position_ids": self.vision_position_ids,
            "und_len": 0,
            "sequence_length": num_vision,
            "vision_sequence_indexes": self.vision_sequence_indexes - self.und_len,
            "vision_mse_loss_indexes": self.vision_mse_loss_indexes - self.und_len,
        }


@adapter_registry.register("cosmos3_base")
class Cosmos3Adapter[TBatch: Cosmos3Batch](
    BaseModelAdapter[Cosmos3OmniTransformer, TBatch]
):
    """Text-to-image generation with the Cosmos3 omni transformer."""

    arch: Literal["cosmos3"] = "cosmos3"
    type: Literal["base"] = "base"

    # One packed sequence per sample: no batch dimension to collate over.
    supports_dense_batching = False

    patch_size: int = 2
    """``latent_patch_size``: the transformer folds 2x2 latent cells per token."""
    vae_scale_factor: int = 16
    latent_channels: int = 48

    hf_model: HfModelLoader[Cosmos3OmniTransformer] = HfModelLoader(
        library="diffusers",
        class_name="Cosmos3OmniTransformer",
        pretrained_model_id="nvidia/Cosmos3-Nano",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )
    peft_lora_config: LoraConfig = LoraConfig(
        # The generation expert only: the und tower is the text encoder.
        target_modules=[
            "self_attn.add_q_proj",
            "self_attn.add_k_proj",
            "self_attn.add_v_proj",
            "self_attn.to_add_out",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
        ],
    )

    def _install_modules(self) -> None:
        # Swap in the meta-safe rotary table and restore the fp32 timestep MLP.
        # Upstream relies on `from_pretrained` honouring `_keep_in_fp32_modules`,
        # which the trainer's meta-init -> `to_empty` -> `dcp.load` path never
        # runs, so `time_embedder` would come back in bf16.
        config = self.transformer.config
        self.transformer.rotary_emb = Cosmos3RotaryEmbedding(
            head_dim=config["head_dim"],
            rope_theta=config["rope_theta"],
            rope_axes_dim=config["rope_axes_dim"],
        )
        self.transformer.time_embedder.float()

    def _prepare_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        # Cosmos3's sigma grid steps by ~7e-4 near sigma=1, well under bf16
        # resolution there; rounding would merge adjacent schedule steps. The
        # transformer keeps `time_embedder` in fp32, so fp32 flows straight in.
        return timestep.to(device=self.device, dtype=torch.float32)

    # ------------------------------ Sequence packing ----------------------------- #

    def _mrope_text_ids(self, num_tokens: int) -> torch.Tensor:
        """Text shares one monotonically increasing id across all three axes."""
        ids = torch.arange(num_tokens, device=self.device, dtype=torch.long)
        return ids.unsqueeze(0).expand(3, -1).contiguous()

    def _mrope_vision_ids(self, grid_h: int, grid_w: int, offset: int) -> torch.Tensor:
        """Single-frame image grid; ``t`` is constant, ``h``/``w`` restart at 0."""
        t = torch.full((grid_h * grid_w,), offset, device=self.device, dtype=torch.long)
        rows = torch.arange(grid_h, device=self.device)
        cols = torch.arange(grid_w, device=self.device)
        h = repeat(rows, "h -> (h w)", w=grid_w)
        w = repeat(cols, "w -> (h w)", h=grid_h)
        return torch.stack([t, h, w], dim=0)

    def _build_pack(self, batch: TBatch) -> Cosmos3Pack:
        input_ids = batch["prompt_embeds"].reshape(-1).to(self.device)
        und_len = int(input_ids.numel())

        height, width = batch["image_size"]
        # The preset resizes to a multiple of `vae_scale_factor * patch_size`, so
        # the grid is exact and nothing needs the upstream pipeline's zero padding.
        grid_h = height // self.vae_scale_factor // self.patch_size
        grid_w = width // self.vae_scale_factor // self.patch_size
        num_vision = grid_h * grid_w

        margin = self.transformer.config["unified_3d_mrope_temporal_modality_margin"]
        vision_position_ids = self._mrope_vision_ids(grid_h, grid_w, und_len + margin)
        vision_indexes = torch.arange(
            und_len, und_len + num_vision, device=self.device, dtype=torch.long
        )
        return Cosmos3Pack(
            input_ids=input_ids,
            text_indexes=torch.arange(und_len, device=self.device, dtype=torch.long),
            position_ids=torch.cat(
                [self._mrope_text_ids(und_len), vision_position_ids], dim=1
            ),
            und_len=und_len,
            sequence_length=und_len + num_vision,
            token_shape=(1, grid_h, grid_w),
            vision_sequence_indexes=vision_indexes,
            # Text-to-image denoises the single latent frame in full.
            vision_mse_loss_indexes=vision_indexes,
            noisy_frame_indexes=torch.zeros(1, device=self.device, dtype=torch.long),
            vision_position_ids=vision_position_ids,
        )

    # -------------------------------- Forward pass ------------------------------- #

    def _predict_velocity(self, batch: TBatch, timestep: torch.Tensor) -> torch.Tensor:
        if "_pack" not in batch:
            batch["_pack"] = self._build_pack(batch)
        pack = batch["_pack"]

        height, width = batch["image_size"]
        latents = self._unpack_latents(
            batch["noisy_latents"],
            h=height // self.vae_scale_factor,
            w=width // self.vae_scale_factor,
        )
        vision_tokens = rearrange(latents, "1 c h w -> 1 c 1 h w").to(self.dtype)

        # The sampler passes sigma in [0, 1] and the transformer multiplies by
        # ``timestep_scale``, so undo that scale here to hand it a model time.
        num_vision = pack.sequence_length - pack.und_len
        vision_timesteps = (
            timestep.reshape(()) / self.transformer.config["timestep_scale"]
        ).expand(num_vision)

        cached = self._begin_forward(batch)
        try:
            preds_vision, _, _ = self.transformer(
                vision_tokens=[vision_tokens],
                vision_token_shapes=[pack.token_shape],
                vision_timesteps=vision_timesteps,
                vision_noisy_frame_indexes=[pack.noisy_frame_indexes],
                return_dict=False,
                **pack.transformer_kwargs(cached),
            )
        finally:
            self._end_forward()

        prediction = rearrange(preds_vision[0], "1 c 1 h w -> 1 c h w")
        return self._pack_latents(prediction)

    def _begin_forward(self, batch: TBatch) -> bool:
        """Arm the KV state for this call; returns whether it replays a cache."""
        return False

    def _end_forward(self) -> None:
        return None


@adapter_registry.register("cosmos3_kv")
class Cosmos3KVAdapter[TBatch: Cosmos3KVBatch](Cosmos3Adapter[TBatch]):
    """Cosmos3 with the text prefix's per-layer K/V reused across denoising steps.

    Bit-exact rather than approximate: the und stream is a function of the
    prompt alone, so replaying its K/V reproduces the uncached velocities.
    """

    type: Literal["kv"] = "kv"

    _kv_state: Cosmos3KVState = PrivateAttr(default_factory=Cosmos3KVState)

    def _install_modules(self) -> None:
        super()._install_modules()
        self._bind_processors()

    def _bind_processors(self) -> None:
        """Point every layer's processor at *this* adapter's state.

        ``_install_modules`` only runs on a fresh load, but ``HfModelLoader``
        caches transformers class-wide: a second adapter sharing the checkpoint
        would otherwise leave the processors bound to the first one's state and
        silently never fill a cache.
        """
        state = self._kv_state
        for index, block in enumerate(self.transformer.layers):
            attn = cast(Cosmos3PackedMoTAttention, cast(Any, block).self_attn)
            processor = attn.get_processor()
            if (
                isinstance(processor, Cosmos3KVAttnProcessor)
                and processor.state is state
            ):
                continue
            # Diffusers' AttentionProcessor union omits its own Cosmos3 processors.
            attn.set_processor(
                Cosmos3KVAttnProcessor(index, state)  # ty: ignore[invalid-argument-type]
            )

    def _begin_forward(self, batch: TBatch) -> bool:
        self._bind_processors()
        reuse = self.use_cache and cache_enabled.get() and not torch.is_grad_enabled()
        if not reuse:
            self._kv_state.mode = None
            self._kv_state.cache = None
            return False
        cache = batch.get("_kv_cache")
        if cache is None:
            cache = batch["_kv_cache"] = Cosmos3UndKVCache()
        self._kv_state.cache = cache
        self._kv_state.mode = "cached" if cache.filled else "extract"
        return cache.filled

    def _end_forward(self) -> None:
        self._kv_state.mode = None
        self._kv_state.cache = None
