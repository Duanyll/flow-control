from typing import Literal, NotRequired, cast

import torch
from diffusers import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import (
    Flux2KVAttnProcessor,
    Flux2KVCache,
    Flux2KVParallelSelfAttnProcessor,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)

from flow_control.adapters.base import adapter_registry
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.model_cache import cache_enabled

from .base import Flux2Adapter, Flux2Batch


class Flux2KVBatch(Flux2Batch):
    _kv_cache: NotRequired[Flux2KVCache]


@adapter_registry.register("flux2_kv")
class Flux2KVAdapter(Flux2Adapter[Flux2KVBatch]):
    """Klein 9B KV checkpoint, with reference-only causal attention."""

    type: Literal["kv"] = "kv"
    supports_dense_batching = False
    hf_model: HfModelLoader[Flux2Transformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="Flux2Transformer2DModel",
        pretrained_model_id="black-forest-labs/FLUX.2-klein-9b-kv",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    def _install_modules(self) -> None:
        # Diffusers' AttentionProcessor union omits its own Flux2 KV processors.
        for block in self.transformer.transformer_blocks:
            cast(Flux2TransformerBlock, block).attn.set_processor(
                Flux2KVAttnProcessor()  # ty: ignore[invalid-argument-type]
            )
        for block in self.transformer.single_transformer_blocks:
            cast(Flux2SingleTransformerBlock, block).attn.set_processor(
                Flux2KVParallelSelfAttnProcessor()  # ty: ignore[invalid-argument-type]
            )

    def _predict_velocity(
        self, batch: Flux2KVBatch, timestep: torch.Tensor
    ) -> torch.Tensor:
        if "_txt_ids" not in batch:
            batch["_txt_ids"] = self.make_text_ids(batch["prompt_embeds"][:1])
        if "_img_ids" not in batch:
            batch["_img_ids"] = self.make_latent_ids(batch["image_size"])
        latents = batch["noisy_latents"]
        img_ids = batch.get("img_ids", batch["_img_ids"])
        reuse = self.use_cache and cache_enabled.get() and not torch.is_grad_enabled()
        kv_cache = batch.get("_kv_cache") if reuse else None
        mode = "cached" if kv_cache is not None else None
        num_ref_tokens = 0
        references = batch.get("reference_latents", [])
        if references and kv_cache is None:
            num_ref_tokens = sum(ref.shape[1] for ref in references)
            latents = torch.cat([*references, latents], dim=1)
            img_ids = torch.cat(
                [self.make_reference_ids(batch["reference_sizes"]), img_ids], dim=1
            )
            # Extract also supplies fixed reference modulation and causal attention.
            # With caching disabled, recompute it each step and discard the KV.
            mode = "extract"
        result = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            guidance=None,
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch.get("txt_ids", batch["_txt_ids"]),
            img_ids=img_ids,
            kv_cache=kv_cache,
            kv_cache_mode=mode,
            num_ref_tokens=num_ref_tokens,
            return_dict=False,
        )
        if reuse and mode == "extract":
            batch["_kv_cache"] = result[1]
        return result[0]
