import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import (
    FluxLoraLoaderMixin,
    PeftAdapterMixin,
    QwenImageLoraLoaderMixin,
)
from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY, LoraBaseMixin
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from pydantic import Field
from safetensors import safe_open
from safetensors.torch import save_file

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.adapters.weights import lora_pairs
from flow_control.scripts.lora import (
    _fuse_transformer,
    _load_external_lora,
    _save_diffusers_lora,
    export_dcp,
    fuse,
    import_lora,
    register_diffusers_lora_loader,
)
from flow_control.utils.hf_model import HfModelLoader


class TinyTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin):
    @register_to_config
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)
        self.proj_extra = torch.nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class TinyFluxTransformer(TinyTransformer):
    pass


class TinyAdapter(BaseModelAdapter[ModelMixin, Batch]):
    """The CLI's view of a training model: native LoRA slots on ``proj``."""

    arch: str = "tiny"
    type: str = "tiny"
    hf_model: HfModelLoader[ModelMixin] = Field(
        default_factory=lambda: HfModelLoader(
            library="custom", class_name="Tiny", pretrained_model_id="tiny"
        )
    )
    lora_target_modules = ("proj",)

    def _predict_velocity(self, batch: Batch, timestep: torch.Tensor) -> torch.Tensor:
        return self.module()(batch["noisy_latents"])


def tiny_adapter(variants: dict[str, Any], seed: int) -> TinyAdapter:
    """A realized tiny adapter; slots not loaded from a ``path`` are random."""
    torch.manual_seed(seed)
    adapter = TinyAdapter(variants=variants, trainable_dtype=torch.float32)
    adapter.hf_model.dtype = torch.float32
    adapter.transformer = TinyTransformer()
    with patch.object(HfModelLoader, "load_model", return_value=True):
        adapter.load_transformer(torch.device("cpu"))
    with torch.no_grad():
        for name, variant in variants.items():
            if "path" in variant:
                continue
            for pair in lora_pairs(adapter.transformer, name).values():
                for parameter in pair.parameters():
                    parameter.normal_()
    return adapter


def make_lora_model(
    seed: int = 0,
    *,
    alpha: int = 4,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    model_type: type[TinyTransformer] = TinyTransformer,
) -> TinyTransformer:
    torch.manual_seed(seed)
    model = model_type()
    model.requires_grad_(False)
    model.add_adapter(
        LoraConfig(
            r=2,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules or ["proj"],
            init_lora_weights=False,
        )
    )
    return model


def adapter_state(model: TinyTransformer) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }


class DiffusersLoraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_diffusers_lora_loader(
            "TinyTransformer",
            QwenImageLoraLoaderMixin,
        )
        register_diffusers_lora_loader(
            "TinyFluxTransformer",
            FluxLoraLoaderMixin,
        )

    def test_export_writes_the_checkpoint_slots_of_one_variant(self) -> None:
        """``lora export --variant`` reads ``app.transformer.*`` slots (an EMA's too)
        into a fresh skeleton and writes exactly them as PEFT tensors; a checkpoint
        missing the variant's slots and a variant with clones are refused."""
        variants = {
            "default": {"type": "lora", "rank": 2, "alpha": 4},
            "ema": {"type": "ema", "of": "default", "decay": 0.9},
        }
        trained = tiny_adapter(variants, seed=1)
        slots = {
            key: value
            for key, value in trained.transformer.state_dict().items()
            if ".lora." in key
        }
        skeleton = tiny_adapter(variants, seed=2)
        plain = tiny_adapter({}, seed=3)
        cloned = tiny_adapter(
            {"default": {"type": "lora", "rank": 2, "extra_trainable": ["proj_extra"]}},
            seed=4,
        )

        def make(adapter: TinyAdapter):  # noqa: ANN202
            return lambda config, *, variants, device="cpu": (
                adapter if variants else plain
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dcp.save(
                {"app": {"transformer": slots}},
                checkpoint_id=root / "ckpt",
                no_dist=True,
            )
            ema_only = {
                key: value for key, value in slots.items() if ".lora.ema." in key
            }
            dcp.save(
                {"app": {"transformer": ema_only}},
                checkpoint_id=root / "ema-only",
                no_dist=True,
            )
            with patch("flow_control.scripts.lora._make_model_adapter", make(skeleton)):
                export_dcp(
                    {},
                    checkpoint_dir=str(root / "ckpt"),
                    output_dir=str(root / "out"),
                    variant="ema",
                )
                with self.assertRaisesRegex(ValueError, "lacks 2 tensors"):
                    export_dcp(
                        {},
                        checkpoint_dir=str(root / "ema-only"),
                        output_dir=str(root / "unused"),
                        variant="default",
                    )
            with (
                patch("flow_control.scripts.lora._make_model_adapter", make(cloned)),
                self.assertRaisesRegex(ValueError, "clones whole modules"),
            ):
                export_dcp(
                    {}, checkpoint_dir=str(root / "ckpt"), output_dir=str(root / "no")
                )
            with safe_open(root / "out/pytorch_lora_weights.safetensors", "pt") as f:
                tensors = {key: f.get_tensor(key) for key in list(f.keys())}
                metadata = json.loads((f.metadata() or {})[LORA_ADAPTER_METADATA_KEY])

        self.assertEqual(
            set(tensors),
            {"transformer.proj.lora_A.weight", "transformer.proj.lora_B.weight"},
        )
        pair = lora_pairs(trained.transformer, "ema")["proj"]
        torch.testing.assert_close(
            tensors["transformer.proj.lora_A.weight"], pair.A.weight
        )
        torch.testing.assert_close(
            tensors["transformer.proj.lora_B.weight"], pair.B.weight
        )
        self.assertEqual(
            (metadata["transformer.r"], metadata["transformer.lora_alpha"]), (2, 4)
        )

    def test_import_writes_a_native_file_a_lora_variant_loads(self) -> None:
        """``lora import`` turns a Diffusers LoRA into the native file a
        ``lora`` variant's ``path`` loads, slot for slot."""
        source = make_lora_model(seed=5)
        expected = adapter_state(source)
        plain = tiny_adapter({}, seed=6)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _save_diffusers_lora(source, str(root / "diffusers"))
            with patch(
                "flow_control.scripts.lora._make_model_adapter", return_value=plain
            ):
                import_lora(
                    {},
                    lora_path=str(root / "diffusers"),
                    output_dir=str(root / "native"),
                    name="style",
                )
            target = tiny_adapter(
                {
                    "default": {
                        "type": "lora",
                        "rank": 2,
                        "alpha": 4,
                        "path": str(root / "native/style.safetensors"),
                    }
                },
                seed=7,
            )

        pair = lora_pairs(target.transformer, "default")["proj"]
        torch.testing.assert_close(pair.A.weight, expected["proj.lora_A.weight"])
        torch.testing.assert_close(pair.B.weight, expected["proj.lora_B.weight"])

    def test_diffusers_export_reloads_through_official_loader(self) -> None:
        source = make_lora_model(seed=1)
        expected = adapter_state(source)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "lora"
            _save_diffusers_lora(source, str(output))
            weight_path = output / "pytorch_lora_weights.safetensors"
            with safe_open(weight_path, framework="pt") as handle:
                self.assertTrue(
                    all(key.startswith("transformer.") for key in list(handle.keys()))
                )

            target = TinyTransformer()
            target.requires_grad_(False)
            _load_external_lora(
                target,
                str(output),
                weight_name=None,
            )

        actual = adapter_state(target)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_fused_transformer_matches_active_adapter(self) -> None:
        model = make_lora_model(seed=11)
        value = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        scale = 0.25
        active = model(value).detach()
        projection = cast(Any, model.proj)
        base = projection.base_layer(value).detach()
        expected = base + scale * (active - base)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fused"
            _fuse_transformer(model, str(output), scale=scale)
            reloaded = TinyTransformer.from_pretrained(output)

        torch.testing.assert_close(reloaded(value), expected)

    def test_qwen_comfy_keys_and_alpha_use_official_converter(self) -> None:
        source = make_lora_model(seed=21)
        source_state = adapter_state(source)
        down = next(
            value
            for key, value in source_state.items()
            if key.endswith("proj.lora_A.weight")
        )
        up = next(
            value
            for key, value in source_state.items()
            if key.endswith("proj.lora_B.weight")
        )
        value = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        expected = source(value).detach()

        with tempfile.TemporaryDirectory() as temp_dir:
            comfy = Path(temp_dir) / "comfy"
            comfy.mkdir()
            save_file(
                {
                    "diffusion_model.proj.lora_down.weight": down,
                    "diffusion_model.proj.lora_up.weight": up,
                    "diffusion_model.proj.alpha": torch.tensor(4.0),
                },
                comfy / "pytorch_lora_weights.safetensors",
            )

            target = TinyTransformer()
            target.requires_grad_(False)
            source_projection = cast(Any, source.proj)
            target_projection = cast(Any, target.proj)
            target_projection.weight.copy_(source_projection.base_layer.weight)
            _load_external_lora(
                target,
                str(comfy),
                weight_name=None,
            )

        torch.testing.assert_close(target(value), expected)

    def test_dora_scale_is_rejected_before_diffusers_discards_it(self) -> None:
        source = make_lora_model(seed=22)
        source_state = adapter_state(source)
        down = next(
            value
            for key, value in source_state.items()
            if key.endswith("proj.lora_A.weight")
        )
        up = next(
            value
            for key, value in source_state.items()
            if key.endswith("proj.lora_B.weight")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_dir = Path(temp_dir) / "dora"
            lora_dir.mkdir()
            save_file(
                {
                    "diffusion_model.proj.lora_down.weight": down,
                    "diffusion_model.proj.lora_up.weight": up,
                    "diffusion_model.proj.alpha": torch.tensor(4.0),
                    "diffusion_model.proj.dora_scale": torch.ones(4),
                },
                lora_dir / "pytorch_lora_weights.safetensors",
            )
            target = TinyTransformer()
            target.requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "lossy conversion"):
                _load_external_lora(target, str(lora_dir), weight_name=None)

    def test_scalar_tensor_metadata_is_saved_as_a_number(self) -> None:
        source = make_lora_model(seed=23)
        peft_config = cast(Any, source.peft_config["default"])
        peft_config.lora_alpha = torch.tensor(4.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "lora"
            _save_diffusers_lora(source, str(output))
            weight_path = output / "pytorch_lora_weights.safetensors"
            with safe_open(weight_path, framework="pt") as handle:
                metadata = handle.metadata()
                assert metadata is not None
                peft_metadata = json.loads(metadata[LORA_ADAPTER_METADATA_KEY])
            alpha = next(
                value
                for key, value in peft_metadata.items()
                if key.endswith("lora_alpha")
            )
            self.assertIsInstance(alpha, (int, float))

            target = TinyTransformer()
            target.requires_grad_(False)
            _load_external_lora(target, str(output), weight_name=None)

        actual = adapter_state(target)
        expected = adapter_state(source)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_external_loader_rejects_partially_consumed_state(self) -> None:
        source = make_lora_model(seed=32)
        packed = LoraBaseMixin.pack_weights(adapter_state(source), "transformer")
        packed["transformer.no_such.lora_A.weight"] = torch.ones(2, 4)
        packed["transformer.no_such.lora_B.weight"] = torch.ones(4, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            external = Path(temp_dir) / "partial"
            external.mkdir()
            save_file(
                packed,
                external / "pytorch_lora_weights.safetensors",
            )
            target = TinyTransformer()
            target.requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "partial conversion"):
                _load_external_lora(target, str(external), weight_name=None)

    def test_flux_loader_preserves_scalar_tensor_alpha(self) -> None:
        source = make_lora_model(seed=28, model_type=TinyFluxTransformer)
        peft_config = cast(Any, source.peft_config["default"])
        peft_config.lora_alpha = torch.tensor(4.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "flux-lora"
            _save_diffusers_lora(source, str(output))
            target = TinyFluxTransformer()
            target.requires_grad_(False)
            _load_external_lora(target, str(output), weight_name=None)

        actual = adapter_state(target)
        expected = adapter_state(source)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        self.assertEqual(float(target.peft_config["default"].lora_alpha), 4.0)

    def test_public_actions_validate_before_loading_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch("flow_control.scripts.lora._make_model_adapter") as load_model:
                with self.assertRaisesRegex(ValueError, "scale must be finite"):
                    fuse(
                        {},
                        lora_path="unused",
                        output_dir=str(root / "fused"),
                        scale=float("nan"),
                    )
                with self.assertRaises(FileNotFoundError):
                    export_dcp(
                        {},
                        checkpoint_dir=str(root / "missing-dcp"),
                        output_dir=str(root / "export"),
                    )
            load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
