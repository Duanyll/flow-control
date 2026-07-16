"""Thin wrapper over the vendored HiDream-O1 transformer.

Adds the surface flow-control's trainer expects on top of the vendored
transformers-style class:

- diffusers-spelling gradient-checkpointing shims
  (``training/mixins/base.py`` checks ``_supports_gradient_checkpointing`` and
  calls ``enable_gradient_checkpointing()``; the transformers spelling would
  silently no-op).
- meta-load safety for the rotary buffers: the vendored code registers
  ``inv_freq`` non-persistent and keeps the text rotary's ``original_inv_freq``
  (the tensor its ``forward`` actually uses) as a *plain attribute*. After the
  trainer's meta-init -> ``to_empty`` -> ``dcp.load`` path those would be
  garbage: non-persistent buffers and plain attributes are absent from the DCP
  seed checkpoint. Re-registering them as persistent buffers makes the seed
  checkpoint (generated from a real CPU load) carry and restore them.
"""

import torch

from flow_control.third_party.hidream_o1.qwen3_vl_transformers import (
    Qwen3VLForConditionalGeneration,
)


class HiDreamO1Transformer(Qwen3VLForConditionalGeneration):
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self._persist_rope_buffers()

    def _compute_rope_tables(self) -> dict[str, torch.Tensor]:
        text_rot = self.model.language_model.rotary_emb
        text_inv_freq, _ = text_rot.rope_init_fn(text_rot.config, None)
        text_inv_freq = text_inv_freq.to(torch.float32).cpu()
        vision_rot = self.model.visual.rotary_pos_emb
        # Mirrors Qwen3VLVisionRotaryEmbedding.__init__ (default theta): the
        # buffer length is len(arange(0, dim, 2)) = dim / 2.
        dim = vision_rot.inv_freq.shape[0] * 2
        vision_inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        return {
            "text.inv_freq": text_inv_freq,
            "text.original_inv_freq": text_inv_freq.clone(),
            "vision.inv_freq": vision_inv_freq,
        }

    def _persist_rope_buffers(self) -> None:
        text_rot = self.model.language_model.rotary_emb
        text_rot.register_buffer("inv_freq", text_rot.inv_freq, persistent=True)
        original_inv_freq = text_rot.original_inv_freq
        del text_rot.original_inv_freq  # plain attribute aliasing inv_freq
        text_rot.register_buffer(
            "original_inv_freq", original_inv_freq.clone(), persistent=True
        )
        vision_rot = self.model.visual.rotary_pos_emb
        vision_rot.register_buffer("inv_freq", vision_rot.inv_freq, persistent=True)

    def to_empty(self, *, device, recurse: bool = True):
        """Re-materialize the rope tables after ``to_empty``.

        After ``to_empty`` the buffers are uninitialized, and the meta-load
        path's earlier ``Module.to(bf16)`` would additionally have quantized
        them (skewing every attention phase by ~1e-2 relative); recomputing
        from the config restores exact float32 values and dtype. Computed here
        rather than in ``__init__``: under from_pretrained the constructor runs
        inside a meta ``DeviceContext`` where even fresh tensors land on meta
        (that path re-initializes the buffers itself and never calls to_empty).
        """
        module = super().to_empty(device=device, recurse=recurse)
        with torch.device("cpu"):
            tables = self._compute_rope_tables()
        text_rot = self.model.language_model.rotary_emb
        vision_rot = self.model.visual.rotary_pos_emb
        for (owner, name), key in [
            ((text_rot, "inv_freq"), "text.inv_freq"),
            ((text_rot, "original_inv_freq"), "text.original_inv_freq"),
            ((vision_rot, "inv_freq"), "vision.inv_freq"),
        ]:
            owner.register_buffer(name, tables[key].to(device), persistent=True)
        return module

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing_disable()
