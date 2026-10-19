"""Ideogram 4: a conditional and an unconditional 9.3B single-stream DiT (asymmetric CFG)."""

from collections.abc import Iterator
from typing import Any, ClassVar, Literal, NotRequired

import torch
import torch.nn.functional as F
from diffusers import Ideogram4Transformer2DModel
from diffusers.models.transformers.transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from einops import rearrange
from pydantic import model_validator

from flow_control.utils.hf_model import Fp8WeightOnlyLoader
from flow_control.utils.types import TorchDType

from .base import BaseModelAdapter, Batch, adapter_registry
from .variants import FullVariant, Variant

IDEOGRAM4_REPO = "ideogram-ai/ideogram-4-fp8"


class Ideogram4Loader(Fp8WeightOnlyLoader[Ideogram4Transformer2DModel]):
    """The fp8 repo keeps the official module names: fused ``attention.qkv`` (and its
    per-row scale) is split into diffusers' ``to_q`` / ``to_k`` / ``to_v``, ``attention.o``
    is ``to_out.0``."""

    library: Literal["diffusers"] = "diffusers"
    class_name: str = "Ideogram4Transformer2DModel"
    pretrained_model_id: str = IDEOGRAM4_REPO
    subfolder: str | None = "transformer"
    dtype: TorchDType = torch.bfloat16

    def _load_with_from_pretrained(self, model_cls, device: torch.device):
        # diffusers wraps `ModelMixin.to`, which ty cannot resolve.
        model: Any = super()._load_with_from_pretrained(model_cls, device)
        # The official package rounds the RoPE table to `dtype` (`model.to(dtype)`
        # before swapping in its fp8 linears); diffusers keeps it fp32. With image
        # positions offset by 65536 the two give materially different images, and
        # the released model's outputs come from the rounded table.
        model.to(dtype=self.dtype)
        return model

    def _convert(
        self, key: str, tensor: torch.Tensor
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if ".attention.qkv." in key:
            parts = zip(("to_q", "to_k", "to_v"), tensor.chunk(3, dim=0), strict=True)
            for name, part in parts:
                yield key.replace(".attention.qkv.", f".attention.{name}."), part
        else:
            yield key.replace(".attention.o.", ".attention.to_out.0."), tensor


class Ideogram4Batch(Batch):
    _ids: NotRequired[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    prompt_embeds: torch.Tensor
    """`[B, L, 13 * 4096]` Qwen3-VL features of the prompt tokens, no padding."""


@adapter_registry.register("ideogram4_base")
class Ideogram4Adapter[TBatch: Ideogram4Batch](
    BaseModelAdapter[Ideogram4Transformer2DModel, TBatch]
):
    """Ideogram 4 adapter: the conditional transformer as the base weights, the
    unconditional one as the full variant ``uncond``.

    The official sampler is asymmetric CFG: the conditional transformer sees
    ``[text | image]`` in one sequence (text slots carry zero latents, image slots
    zero text features) and the unconditional transformer sees the image tokens
    alone with zeroed text features. Which inputs to build follows the active
    variant's :meth:`origin`, so a config expresses the official guidance as
    ``{"type": "cfg", "negative": {"type": "model", "variant": "uncond"},
    "negative_condition": "positive"}``.

    Conventions translated in :meth:`_predict_velocity`: the model's time runs
    0 (noise) -> 1 (data) and its velocity points at the data, so it gets
    ``1 - sigma`` and returns the negated output; the packed latent channels are
    ``(ph pw c)``-ordered (``_pack_latents`` and the VAE constants use the
    framework's ``(c ph pw)``), permuted around the forward. Image positions are
    ``(0, h, w) + IMAGE_POSITION_OFFSET``, text positions ``arange(L)`` on every
    mRoPE axis; one segment, so the sequence is never padded.
    """

    arch: Literal["ideogram4"] = "ideogram4"
    type: Literal["base"] = "base"

    latent_channels: int = 32

    hf_model: Ideogram4Loader = Ideogram4Loader()
    variants: dict[str, Variant] = {
        "uncond": FullVariant(
            hf_model=Ideogram4Loader(subfolder="unconditional_transformer")
        )
    }
    uncond_variant: ClassVar[str] = "uncond"
    """Variants rooted here run image-only with zeroed text features."""
    lora_target_modules = ("to_q", "to_k", "to_v", "to_out.0")

    @model_validator(mode="before")
    @classmethod
    def _fp8_loaders(cls, data: Any) -> Any:
        """A full variant's `hf_model` written in a config is this repo's fp8 loader."""
        if not isinstance(data, dict) or not isinstance(data.get("variants"), dict):
            return data
        variants = {}
        for name, variant in data["variants"].items():
            if isinstance(variant, dict) and isinstance(variant.get("hf_model"), dict):
                variant = {
                    **variant,
                    "hf_model": Ideogram4Loader(**variant["hf_model"]),
                }
            variants[name] = variant
        return {**data, "variants": variants}

    supports_dense_batching = True
    dense_batch_fields = ("image_size", "noisy_latents", "prompt_embeds")
    shared_cache_fields = ("_ids",)

    def _prepare_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        # Near sigma = 1 the model time 1 - sigma underflows bf16 resolution.
        return timestep.to(device=self.device, dtype=torch.float32)

    def _make_ids(
        self, batch: TBatch, text_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(position_ids, segment_ids, indicator)`` of the ``[text | image]`` sequence, batch 1."""
        h, w = batch["image_size"]
        scale = self.vae_scale_factor * self.patch_size
        grid = torch.cartesian_prod(torch.arange(h // scale), torch.arange(w // scale))
        image = F.pad(grid, (1, 0)) + IMAGE_POSITION_OFFSET
        text = torch.arange(text_len)[:, None].expand(-1, 3)
        indicator = torch.cat(
            [
                torch.full((text_len,), LLM_TOKEN_INDICATOR),
                torch.full((len(image),), OUTPUT_IMAGE_INDICATOR),
            ]
        )
        position_ids = torch.cat([text, image])[None].to(self.device)
        indicator = indicator[None].to(self.device)
        return position_ids, torch.ones_like(indicator), indicator

    def _predict_velocity(self, batch: TBatch, timestep: torch.Tensor) -> torch.Tensor:
        patch = {"ph": self.patch_size, "pw": self.patch_size}
        x = rearrange(batch["noisy_latents"], "b n (c ph pw) -> b n (ph pw c)", **patch)
        b, n, d = x.shape
        image_only = self.origin(self.active_variant) == self.uncond_variant
        text_len = 0 if image_only else batch["prompt_embeds"].shape[1]
        if "_ids" not in batch:
            batch["_ids"] = self._make_ids(batch, text_len)
        position_ids, segment_ids, indicator = (
            ids.expand(b, *ids.shape[1:]) for ids in batch["_ids"]
        )
        features = x.new_zeros(
            b, text_len + n, self.transformer.config["llm_features_dim"]
        )
        if text_len:
            features[:, :text_len] = batch["prompt_embeds"]
            x = torch.cat([x.new_zeros(b, text_len, d), x], dim=1)
        out = self.module()(
            hidden_states=x,
            timestep=(1 - timestep).to(self.dtype),
            encoder_hidden_states=features,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
            return_dict=False,
        )[0]
        return rearrange(-out[:, text_len:], "b n (ph pw c) -> b n (c ph pw)", **patch)


if __name__ == "__main__":
    from types import SimpleNamespace

    from rich import print

    adapter = Ideogram4Adapter()
    assert adapter.origin("uncond") == "uncond" and adapter.origin("default") == "base"
    loader = Ideogram4Loader()
    weight = torch.arange(6.0).view(3, 2)
    converted = dict(loader._convert("layers.0.attention.qkv.weight", weight))
    assert [k.split(".")[-2] for k in converted] == ["to_q", "to_k", "to_v"]
    assert torch.equal(converted["layers.0.attention.to_v.weight"], weight[2:])
    assert dict(loader._convert("layers.0.attention.o.weight", weight)) == {
        "layers.0.attention.to_out.0.weight": weight
    }
    batch: Ideogram4Batch = {
        "image_size": (32, 48),
        "noisy_latents": torch.zeros(1, 6, 128),
        "clean_latents": torch.zeros(1, 6, 128),
        "prompt_embeds": torch.zeros(1, 2, 8),
    }
    # `_make_ids` only needs the device; no transformer is loaded here.
    adapter.hf_model.model = SimpleNamespace(device=torch.device("cpu"))
    position_ids, segment_ids, indicator = adapter._make_ids(batch, 2)
    assert position_ids.shape == (1, 8, 3) and segment_ids.all()
    assert indicator[0].tolist() == [3, 3, 2, 2, 2, 2, 2, 2]
    assert position_ids[0, :2].tolist() == [[0, 0, 0], [1, 1, 1]]
    assert position_ids[0, 2:].sub(IMAGE_POSITION_OFFSET).tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 2],
    ]
    print("[green]ideogram4 key conversion and sequence layout ok[/green]")
