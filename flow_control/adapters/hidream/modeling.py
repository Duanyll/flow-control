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
- load safety for the same buffers: transformers 5.x ``from_pretrained``
  builds the model on meta and only re-initializes missing buffers its generic
  ``_init_weights`` recognizes. It restores the text rotary but not the vendored
  vision rotary, whose now-persistent ``inv_freq`` (missing from the checkpoint)
  was left as uninitialized memory -- and then copied into every DCP seed.
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

    def _restore_rope_tables(self) -> None:
        """Recompute the rope buffers in exact float32, each on its own device.

        Computed after loading rather than in ``__init__``: both load paths
        construct the model under meta, where even fresh tensors land on meta.
        """
        with torch.device("cpu"):
            tables = self._compute_rope_tables()
        text_rot = self.model.language_model.rotary_emb
        vision_rot = self.model.visual.rotary_pos_emb
        for (owner, name), key in [
            ((text_rot, "inv_freq"), "text.inv_freq"),
            ((text_rot, "original_inv_freq"), "text.original_inv_freq"),
            ((vision_rot, "inv_freq"), "vision.inv_freq"),
        ]:
            device = getattr(owner, name).device
            owner.register_buffer(name, tables[key].to(device), persistent=True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model._restore_rope_tables()
        return model

    def to_empty(self, *, device, recurse: bool = True):
        """Re-materialize the rope tables after ``to_empty``.

        After ``to_empty`` the buffers are uninitialized, and the trainer's
        meta-init path (``HfModelLoader._load_model_on_meta``) would additionally
        have quantized them to bf16 (skewing every attention phase by ~1e-2
        relative); recomputing from the config restores exact float32 values.
        """
        module = super().to_empty(device=device, recurse=recurse)
        self._restore_rope_tables()
        return module

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing_disable()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from rich import print
    from safetensors.torch import save_file
    from transformers import AutoConfig

    # A tiny random HiDream-O1 saved like the real checkpoint (no rope buffers),
    # reloaded through both paths that construct the model on meta.
    config = AutoConfig.from_pretrained("HiDream-ai/HiDream-O1-Image")
    config.text_config.update(
        {
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
        }
    )
    config.vision_config.update(
        {
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [],
        }
    )
    model = HiDreamO1Transformer(config)
    expected = model._compute_rope_tables()

    with tempfile.TemporaryDirectory() as tmp:
        config.save_pretrained(tmp)
        state = {k: v for k, v in model.state_dict().items() if "inv_freq" not in k}
        save_file(state, Path(tmp) / "model.safetensors")
        loaded = HiDreamO1Transformer.from_pretrained(tmp, dtype=torch.bfloat16)

    with torch.device("meta"):
        meta = HiDreamO1Transformer(config)
    # As the meta-load path does. transformers wraps `to()` in functools.wraps,
    # so ty reads the decorated signature as unbound.
    meta.to(dtype=torch.bfloat16)  # ty: ignore[missing-argument]
    meta.to_empty(device=torch.device("cpu"))

    for path, m in (("from_pretrained", loaded), ("meta + to_empty", meta)):
        text_rot = m.model.language_model.rotary_emb
        got = {
            "text.inv_freq": text_rot.inv_freq,
            "text.original_inv_freq": text_rot.original_inv_freq,
            "vision.inv_freq": m.model.visual.rotary_pos_emb.inv_freq,
        }
        for key, value in expected.items():
            assert torch.equal(got[key], value), f"{path}: {key} = {got[key]}"
        print(f"[green]{path}: rope tables exact float32[/]")
