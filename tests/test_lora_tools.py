import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from pydantic import ConfigDict
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from flow_control.scripts.lora import (
    _copy_lora_adapter,
    _DcpLoraState,
    _fuse_transformer,
    _load_dcp_lora,
    _load_external_lora,
    _save_dcp_adapter,
    _save_diffusers_lora,
    export_dcp,
    fuse,
    register_diffusers_lora_loader,
)
from flow_control.training.mixins.dcp import DcpMixin


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


class PartialCheckpointState(DcpMixin):
    """Minimal trainer-shaped state used to lock in partial DCP semantics."""

    transformer: Any
    optimizer_marker: torch.Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def state_dict(self) -> dict[str, Any]:
        options = StateDictOptions(strict=False, ignore_frozen_params=True)
        return {
            "transformer": get_model_state_dict(self.transformer, options=options),
            "optimizer": {"marker": self.optimizer_marker},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        options = StateDictOptions(strict=False, ignore_frozen_params=True)
        set_model_state_dict(
            self.transformer,
            state_dict["transformer"],
            options=options,
        )
        self.optimizer_marker.copy_(state_dict["optimizer"]["marker"])


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

    def test_frozen_inference_adapter_can_be_normalized(self) -> None:
        source = make_lora_model(seed=31)
        cast(Any, source.peft_config["default"]).inference_mode = True
        with tempfile.TemporaryDirectory() as temp_dir:
            external = Path(temp_dir) / "external"
            normalized = Path(temp_dir) / "normalized"
            _save_diffusers_lora(source, str(external))

            target = TinyTransformer()
            target.requires_grad_(False)
            _load_external_lora(target, str(external), weight_name=None)
            self.assertFalse(
                any(parameter.requires_grad for parameter in target.parameters())
            )
            _save_diffusers_lora(target, str(normalized))

        self.assertEqual(adapter_state(target).keys(), adapter_state(source).keys())

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

    def test_import_rejects_different_target_layouts(
        self,
    ) -> None:
        source = make_lora_model(seed=12)

        target = TinyTransformer()
        target.requires_grad_(False)
        target.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["proj", "proj_extra"],
                init_lora_weights=False,
            ),
            adapter_name="default",
        )
        target.add_adapter(source.peft_config["default"], adapter_name="imported")
        source_state = get_peft_model_state_dict(source)
        set_peft_model_state_dict(target, source_state, adapter_name="imported")
        with self.assertRaisesRegex(ValueError, "target layouts differ"):
            _copy_lora_adapter(
                target,
                source_name="imported",
                target_name="default",
            )

    def test_import_reparameterizes_different_scaling(self) -> None:
        source = make_lora_model(seed=13, alpha=2)
        target = make_lora_model(seed=14, alpha=4, dropout=0.25)
        target.add_adapter(source.peft_config["default"], adapter_name="imported")
        set_peft_model_state_dict(
            target,
            get_peft_model_state_dict(source),
            adapter_name="imported",
        )
        value = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        target.eval()
        target.set_adapters("imported")
        expected = target(value).detach()

        _copy_lora_adapter(
            target,
            source_name="imported",
            target_name="default",
        )
        torch.testing.assert_close(target(value), expected)

    def test_transformer_only_dcp_roundtrip(self) -> None:
        source = make_lora_model(seed=2)
        expected = adapter_state(source)
        adapter: Any = SimpleNamespace(transformer=source)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "adapter-dcp"
            _save_dcp_adapter(adapter, str(output))

            target = make_lora_model(seed=99)
            _load_dcp_lora(target, str(output), "current")

        actual = adapter_state(target)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_generic_loader_accepts_transformer_only_partial_dcp(self) -> None:
        source = make_lora_model(seed=29)
        expected = adapter_state(source)
        adapter: Any = SimpleNamespace(transformer=source)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "partial-dcp"
            _save_dcp_adapter(adapter, str(output))
            target = make_lora_model(seed=30)
            state = PartialCheckpointState(
                transformer=target,
                optimizer_marker=torch.tensor(123.0),
            )
            state.load_dcp_checkpoint(str(output))

        self.assertEqual(state.optimizer_marker.item(), 123.0)
        actual = adapter_state(target)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_transformer_only_dcp_roundtrip_with_multiple_targets(self) -> None:
        source = make_lora_model(
            seed=25,
            target_modules=["proj", "proj_extra"],
        )
        expected = adapter_state(source)
        adapter: Any = SimpleNamespace(transformer=source)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "adapter-dcp"
            _save_dcp_adapter(adapter, str(output))
            target = make_lora_model(
                seed=26,
                target_modules=["proj", "proj_extra"],
            )
            _load_dcp_lora(target, str(output), "current")

        actual = adapter_state(target)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_ema_selections_use_optimizer_shadow(self) -> None:
        for selection, expected_value in (("ema", 0.75), ("ema_old", 0.25)):
            with self.subTest(selection=selection):
                source = make_lora_model(seed=3)
                stateful = _DcpLoraState(source, selection)  # type: ignore[arg-type]
                assert stateful.ema_optimizer is not None
                for parameter_state in stateful.ema_optimizer.state.values():
                    parameter_state["ema_buffer"].fill_(expected_value)

                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint = Path(temp_dir) / "training-dcp"
                    dcp.save({"app": stateful}, checkpoint_id=checkpoint, no_dist=True)
                    target = make_lora_model(seed=4)
                    _load_dcp_lora(
                        target,
                        str(checkpoint),
                        selection,  # type: ignore[arg-type]
                    )

                for value in adapter_state(target).values():
                    torch.testing.assert_close(
                        value, torch.full_like(value, expected_value)
                    )

    def test_dcp_missing_configured_lora_tensors_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "wrong-transformer-state"
            dcp.save(
                {"app": {"transformer": {"proj.weight": torch.ones(4, 4)}}},
                checkpoint_id=checkpoint,
                no_dist=True,
            )
            target = make_lora_model(seed=15)
            with self.assertRaises(CheckpointException):
                _load_dcp_lora(target, str(checkpoint), "current")

    def test_diffusers_output_requires_safetensors_suffix(self) -> None:
        model = make_lora_model(seed=16)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "bad-output"
            with self.assertRaisesRegex(ValueError, "must end with '.safetensors'"):
                _save_diffusers_lora(
                    model,
                    str(output),
                    weight_name="adapter.bin",
                )
            self.assertFalse(output.exists())

    def test_export_rejects_non_lora_trainable_parameters(self) -> None:
        model = make_lora_model(seed=18)
        model.proj_extra.weight.requires_grad_(True)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "invalid-export"
            with self.assertRaisesRegex(ValueError, "also has trainable parameters"):
                _save_diffusers_lora(model, str(output))
            self.assertFalse(output.exists())

    def test_missing_ema_is_rejected_before_partial_load(self) -> None:
        source = make_lora_model(seed=5)
        options = StateDictOptions(strict=False, ignore_frozen_params=True)
        state = get_model_state_dict(source, options=options)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "current-only"
            dcp.save(
                {"app": {"transformer": state}},
                checkpoint_id=checkpoint,
                no_dist=True,
            )
            target = make_lora_model(seed=6)
            with self.assertRaisesRegex(ValueError, "no app.optim_ema state"):
                _load_dcp_lora(target, str(checkpoint), "ema")

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
