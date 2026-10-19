import os
import tempfile
import unittest
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import patch

import torch
from diffusers import ModelMixin
from pydantic import Field, PrivateAttr
from torch import nn
from torch.nn import functional as F

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.adapters.variants import (
    LoraLinear,
    LoraVariant,
    VariantModule,
    save_lora,
)
from flow_control.utils.hf_model import HfModelLoader


class Block(nn.Module):
    _gradient_checkpointing_func: Callable[..., torch.Tensor]  # set by diffusers

    def __init__(self, width: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return self._gradient_checkpointing_func(self.run, x)
        return self.run(x)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + torch.tanh(self.to_q(x)))


class TinyTransformer(ModelMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.embed = nn.Linear(2, width)
        self.blocks = nn.ModuleList([Block(width), Block(width)])
        self.out = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + t.view(-1, 1, 1)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class TinyAdapter(BaseModelAdapter[ModelMixin, Batch]):
    arch: str = "tiny"
    type: str = "tiny"
    hf_model: HfModelLoader[ModelMixin] = Field(
        default_factory=lambda: HfModelLoader(
            library="custom", class_name="Tiny", pretrained_model_id="tiny"
        )
    )
    lora_target_modules = ("to_q",)
    _misses: list[str] = PrivateAttr(default_factory=list)

    def _predict_velocity(self, batch: Batch, timestep: torch.Tensor) -> torch.Tensor:
        data = cast(dict[str, Any], batch)
        if "_ids" not in data:  # an adapter cache: must be rebuilt once per variant
            data["_ids"] = torch.tensor(float(len(self._misses)))
            self._misses.append(self._active.get())
        return self.module()(batch["noisy_latents"], timestep)


def load(
    variants: dict[str, Any],
    *,
    trainable: bool = False,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> TinyAdapter:
    """Load a tiny adapter the way `load_transformer` does, with the HF loader faked."""
    torch.manual_seed(seed)
    adapter = TinyAdapter(variants=variants, trainable=trainable, trainable_dtype=dtype)
    adapter.hf_model.dtype = dtype
    adapter.transformer = TinyTransformer().to(dtype).requires_grad_(trainable)
    with patch.object(HfModelLoader, "load_model", return_value=True):
        adapter.load_transformer(torch.device("cpu"))
    return adapter


def perturb(tensors: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for tensor in tensors:
            tensor.normal_()


def row(seed: int = 0) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 5, 2, generator=generator)
    return {"image_size": (8, 8), "clean_latents": latents, "noisy_latents": latents}


class VariantTest(unittest.TestCase):
    def test_lora_slots_follow_active_variant(self) -> None:
        """Replaces PEFT's set_adapter traversal: the forward reads the ContextVar."""
        adapter = load(
            {
                "default": {
                    "type": "lora",
                    "rank": 2,
                    "alpha": 4,
                    "bias": True,
                    "extra_trainable": ["norm"],
                },
                "guide": {"type": "lora", "rank": 3, "trainable": False},
                "ref": {"type": "alias", "of": "base"},
                "uncond": {"type": "full"},
            }
        )
        transformer = cast(Any, adapter.transformer)
        for name in ("default", "guide"):
            perturb(adapter._weights(name))
        linear = transformer.blocks[0].to_q
        norm = transformer.blocks[0].norm
        self.assertIsInstance(linear, LoraLinear)
        self.assertIsInstance(norm, VariantModule)
        h = torch.randn(1, 5, 4)
        plain = F.linear(h, linear.weight, linear.bias)
        expected = {
            "default": plain
            + F.linear(
                F.linear(h, linear.lora["default"].A.weight),
                linear.lora["default"].B.weight,
                linear.lora["default"].B.bias,
            )
            * 2.0,
            "guide": plain
            + F.linear(h, linear.lora["guide"].A.weight)
            @ linear.lora["guide"].B.weight.T,
            "ref": plain,
            "base": plain,
            "uncond": plain,
        }
        for name, value in expected.items():
            with adapter.use_variant(name):
                torch.testing.assert_close(linear(h), value)
                clone = norm.clones["default"] if name == "default" else norm.base
                torch.testing.assert_close(norm(h), clone(h))
        self.assertEqual(adapter._active.get(), "default")

        # "uncond" runs its own copy of the base weights; the base is untouched.
        x, t = row()["noisy_latents"], torch.tensor([0.5])
        with adapter.use_variant("ref"):
            reference = adapter.module()(x, t)
        with adapter.use_variant("uncond"):
            self.assertIs(adapter.module(), transformer.variants["uncond"])
            torch.testing.assert_close(adapter.module()(x, t), reference)
            perturb(list(adapter.module().parameters()))
            self.assertFalse(torch.allclose(adapter.module()(x, t), reference))
        with adapter.use_variant("ref"):
            torch.testing.assert_close(adapter.module()(x, t), reference)

        # Resolution, nesting and restoration.
        self.assertEqual(adapter.resolve("ref"), "base")
        self.assertEqual(adapter.resolve("default"), "default")
        self.assertEqual(TinyAdapter().resolve("default"), "base")
        with self.assertRaisesRegex(ValueError, "guide"):
            adapter.resolve("missing")
        with adapter.use_variant("guide"):
            with adapter.use_variant("ref"), adapter.use_variant(None):
                self.assertEqual(adapter._active.get(), "base")
            self.assertEqual(adapter._active.get(), "guide")
        self.assertEqual(adapter._active.get(), "default")

        # An aliased "default" is what runs when nothing selects a variant.
        aliased = load(
            {
                "policy": {"type": "lora", "rank": 2},
                "default": {"type": "alias", "of": "policy"},
            }
        )
        perturb(aliased._weights("policy"))
        with aliased.use_variant("base"):
            plain = aliased.module()(x, t)
        with aliased.use_variant("policy"):
            expected = aliased.module()(x, t)
        torch.testing.assert_close(aliased.module()(x, t), expected)
        self.assertFalse(torch.allclose(expected, plain))

        # Trainability and the state-dict layout the checkpoints rely on.
        trainable = {id(p) for p in adapter.trainable_parameters()}
        self.assertEqual(trainable, {id(p) for p in adapter._weights("default")})
        owned = {id(p) for m in adapter.variant_modules() for p in m.parameters()}
        for name in ("default", "guide", "uncond"):
            self.assertTrue(owned.issuperset(id(p) for p in adapter._weights(name)))
        self.assertTrue(owned.isdisjoint(id(p) for p in adapter._weights("base")))
        keys = set(transformer.state_dict())
        self.assertTrue(
            keys.issuperset(
                {
                    "blocks.0.to_q.weight",
                    "blocks.0.to_q.lora.default.A.weight",
                    "blocks.0.to_q.lora.default.B.bias",
                    "blocks.0.to_q.lora.guide.B.weight",
                    "blocks.0.norm.base.weight",
                    "blocks.0.norm.clones.default.weight",
                    "variants.uncond.blocks.0.to_q.weight",
                }
            )
        )
        self.assertNotIn("blocks.0.to_q.lora.guide.B.bias", keys)

    def test_ema_tracks_source_with_warmup(self) -> None:
        """EMA is a lerp on resident fp32 slots: no optimizer, no weight swap."""
        adapter = load(
            {
                "default": {"type": "lora", "rank": 2, "extra_trainable": ["norm"]},
                "ema": {
                    "type": "ema",
                    "decay": 0.5,
                    "warmup": {"type": "standard"},
                },
                "old": {"type": "ema", "decay": 0.9, "every": "epoch"},
            }
        )
        source = adapter._weights("default")
        ema, old = adapter._weights("ema"), adapter._weights("old")
        self.assertEqual(len(ema), len(source))
        for slot, origin in zip(ema, source, strict=True):
            torch.testing.assert_close(slot, origin)
            self.assertEqual(slot.dtype, torch.float32)
            self.assertFalse(slot.requires_grad)
        initial = [slot.clone() for slot in ema]
        perturb(source)
        adapter.update_variants("step")
        decay = min(1 / 10, 0.5)
        for slot, start, origin in zip(ema, initial, source, strict=True):
            torch.testing.assert_close(slot, start + (1 - decay) * (origin - start))
        for slot, start in zip(old, initial, strict=True):
            torch.testing.assert_close(slot, start)
        adapter.update_variants("epoch")
        for slot, start, origin in zip(old, initial, source, strict=True):
            torch.testing.assert_close(slot, start + 0.1 * (origin - start))
        counts = adapter.transformer.state_dict()
        self.assertEqual(int(counts["variant_counts.ema"]), 1)
        self.assertEqual(int(counts["variant_counts.old"]), 1)

        # fp32 EMA weights (a full copy of a trainable base, or LoRA slots and
        # clones) run in the bf16 model dtype.
        ema_spec = {"type": "ema", "decay": 0.0}
        for default, trainable in (
            ({"type": "alias", "of": "base"}, True),
            ({"type": "lora", "rank": 2, "extra_trainable": ["norm"]}, False),
        ):
            adapter = load(
                {"default": default, "ema": ema_spec},
                trainable=trainable,
                dtype=torch.bfloat16,
            )
            source = adapter._weights(adapter.resolve("default"))
            ema = adapter._weights("ema")
            self.assertEqual({p.dtype for p in source}, {torch.bfloat16})
            self.assertEqual({p.dtype for p in ema}, {torch.float32})
            self.assertEqual(
                {id(p) for p in adapter.trainable_parameters()},
                {id(p) for p in source},
            )
            perturb(source)
            adapter.update_variants("step")
            for slot, origin in zip(ema, source, strict=True):
                torch.testing.assert_close(slot, origin.float())
            with torch.no_grad(), adapter.use_variant("ema"):
                (velocity,) = adapter.predict_velocity_batched(
                    [row()], [torch.tensor([0.5])]
                )
            with torch.no_grad():
                (reference,) = adapter.predict_velocity_batched(
                    [row()], [torch.tensor([0.5])]
                )
            torch.testing.assert_close(velocity, reference, atol=0.1, rtol=0.1)

    def test_checkpoint_recompute_keeps_forward_variant(self) -> None:
        """Backward runs after `use_variant` exits; recompute must still see "guide"."""
        reference: list[torch.Tensor] = []
        for checkpointing in (False, True):
            adapter = load(
                {
                    "default": {"type": "lora", "rank": 2},
                    "guide": {"type": "lora", "rank": 2},
                }
            )
            for name in ("default", "guide"):
                perturb(adapter._weights(name))
            if checkpointing:
                adapter.enable_gradient_checkpointing()
            with adapter.use_variant("guide"):
                (velocity,) = adapter.predict_velocity_batched(
                    [row()], [torch.tensor([0.5])]
                )
            velocity.square().sum().backward()
            grads = [cast(torch.Tensor, p.grad) for p in adapter._weights("guide")]
            self.assertTrue(all(p.grad is None for p in adapter._weights("default")))
            if not checkpointing:
                reference = grads
                continue
            self.assertTrue(
                cast(Any, adapter.transformer).blocks[0].gradient_checkpointing
            )
            for grad, expected in zip(grads, reference, strict=True):
                torch.testing.assert_close(grad, expected)

    def test_row_cache_is_per_variant(self) -> None:
        """Rows persist across steps; two variants on one row keep separate caches."""
        adapter = load(
            {
                "default": {"type": "lora", "rank": 2},
                "guide": {"type": "lora", "rank": 2},
                "ref": {"type": "alias", "of": "base"},
            }
        )
        sample = row()
        with torch.no_grad():
            for name in ("default", "guide", "default", "ref", "base", "guide"):
                with adapter.use_variant(name):
                    adapter.predict_velocity_batched([sample], [torch.tensor([0.5])])
        self.assertEqual(adapter._misses, ["default", "guide", "base"])
        caches = cast(dict[str, Any], sample)["_cache"]
        self.assertEqual(set(caches), {"default", "guide", "base"})
        self.assertEqual(
            [float(caches[name]["_ids"]) for name in ("default", "guide", "base")],
            [0.0, 1.0, 2.0],
        )
        # Under autograd nothing is reused or written back.
        adapter.predict_velocity_batched([sample], [torch.tensor([0.5])])
        self.assertEqual(adapter._misses, ["default", "guide", "base", "default"])
        self.assertEqual(float(caches["default"]["_ids"]), 0.0)
        self.assertFalse(any(key.startswith("_") for key in sample if key != "_cache"))

    def test_native_file_round_trip(self) -> None:
        """`LoraVariant.path` reloads exactly what `save_lora` wrote, or refuses."""
        spec: dict[str, Any] = {"type": "lora", "rank": 2, "alpha": 4, "bias": True}
        trained = load({"default": spec})
        perturb(trained._weights("default"))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "default.safetensors")
            variant = cast(LoraVariant, trained.variants["default"])
            save_lora(trained.transformer, "default", variant, path)
            guide = load({"guide": {**spec, "path": path, "trainable": False}}, seed=1)
            guide.transformer.load_state_dict(
                {
                    k: v
                    for k, v in trained.transformer.state_dict().items()
                    if ".lora." not in k
                },
                strict=False,
            )
            x, t = row()["noisy_latents"], torch.tensor([0.5])
            with trained.use_variant("default"):
                expected = trained.module()(x, t)
            with guide.use_variant("guide"):
                torch.testing.assert_close(guide.module()(x, t), expected)
            self.assertFalse(any(p.requires_grad for p in guide._weights("guide")))
            for wrong in (
                {**spec, "rank": 3, "path": path},
                {**spec, "path": path, "target_modules": ["embed"]},
            ):
                with self.assertRaisesRegex(ValueError, "does not fit"):
                    load({"guide": wrong})


if __name__ == "__main__":
    unittest.main()
