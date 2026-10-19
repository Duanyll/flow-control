"""Offline LoRA conversion backed by Diffusers and PEFT; training itself uses native slots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dcp
from diffusers.loaders import (
    Flux2LoraLoaderMixin,
    FluxLoraLoaderMixin,
    Krea2LoraLoaderMixin,
    QwenImageLoraLoaderMixin,
    SD3LoraLoaderMixin,
    ZImageLoraLoaderMixin,
)
from diffusers.loaders.lora_base import (
    LORA_ADAPTER_METADATA_KEY,
    LORA_WEIGHT_NAME_SAFE,
    LoraBaseMixin,
    _fetch_state_dict,
)
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from flow_control.adapters import parse_model_adapter
from flow_control.adapters.base import BaseModelAdapter
from flow_control.adapters.variants import EmaVariant, LoraVariant
from flow_control.adapters.weights import VariantModule, lora_metadata, lora_pairs
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

_DIFFUSERS_LORA_LOADERS: dict[str, type[LoraBaseMixin]] = {}


def register_diffusers_lora_loader(
    transformer_class_name: str,
    loader_mixin: type[LoraBaseMixin],
) -> None:
    """Register an official pipeline LoRA converter for an adapter plugin."""
    existing = _DIFFUSERS_LORA_LOADERS.get(transformer_class_name)
    if existing is not None and existing is not loader_mixin:
        raise ValueError(
            f"A Diffusers LoRA loader is already registered for "
            f"{transformer_class_name}: {existing.__name__}"
        )
    _DIFFUSERS_LORA_LOADERS[transformer_class_name] = loader_mixin


for _model_class_name, _loader_mixin in (
    ("FluxTransformer2DModel", FluxLoraLoaderMixin),
    ("Flux2Transformer2DModel", Flux2LoraLoaderMixin),
    ("Krea2Transformer2DModel", Krea2LoraLoaderMixin),
    ("QwenImageTransformer2DModel", QwenImageLoraLoaderMixin),
    ("SD3Transformer2DModel", SD3LoraLoaderMixin),
    ("ZImageTransformer2DModel", ZImageLoraLoaderMixin),
):
    register_diffusers_lora_loader(_model_class_name, _loader_mixin)


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        raise TypeError(
            f"Only scalar tensors can be encoded in LoRA metadata, got {value.shape}."
        )
    return str(value)


def _require_new_output(path: str) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(
            f"Output path already exists: {output}. Choose a new path to avoid "
            "mixing or overwriting model files."
        )
    return output


def _require_dcp_checkpoint(path: str) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_dir() or not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(
            f"Not a DCP checkpoint: {checkpoint} (missing directory or .metadata)."
        )
    return checkpoint


def _validate_output_weight_name(weight_name: str) -> None:
    if not weight_name or Path(weight_name).name != weight_name:
        raise ValueError(
            f"Output weight_name must be a file name, got {weight_name!r}."
        )
    if not weight_name.endswith(".safetensors"):
        raise ValueError(
            "Diffusers LoRA output uses safetensors; output weight_name must end "
            f"with '.safetensors', got {weight_name!r}."
        )


def _validate_fuse_scale(scale: float) -> None:
    if not math.isfinite(scale):
        raise ValueError(f"LoRA fuse scale must be finite, got {scale}.")


def _make_model_adapter(
    config: dict[str, Any], *, variants: bool, device: str = "cpu"
) -> BaseModelAdapter:
    """The config's transformer; ``variants=False`` is a plain copy PEFT can wrap."""
    raw_model = config.get("model")
    if not isinstance(raw_model, dict):
        raise ValueError("The config must contain a model object.")
    adapter = parse_model_adapter(
        raw_model if variants else {**raw_model, "variants": {}, "trainable": False}
    )
    # A CPU-only converter must not inherit a serving/training CUDA device map.
    adapter.hf_model.device_memory_distribution = None
    adapter.load_transformer(torch.device(device))
    return adapter


def _lora_slots(adapter: BaseModelAdapter, name: str) -> tuple[str, LoraVariant]:
    """The resolved slot name and the LoRA config shaping it (an EMA shares its source's)."""
    name = adapter.resolve(name)
    variant = adapter.variants.get(name)
    while isinstance(variant, EmaVariant):
        variant = adapter.variants.get(adapter.resolve(variant.of))
    if not isinstance(variant, LoraVariant):
        raise ValueError(
            f"Variant {name!r} is not realized by LoRA slots; Diffusers LoRA cannot "
            "represent it."
        )
    clones = [
        fqn
        for fqn, module in adapter.transformer.named_modules()
        if isinstance(module, VariantModule) and name in module.clones
    ]
    if clones:
        raise ValueError(
            f"Variant {name!r} clones whole modules (extra_trainable) {clones[:5]}; "
            "Diffusers LoRA cannot represent them."
        )
    return name, variant


def _load_dcp_slots(transformer: torch.nn.Module, checkpoint: Path, name: str) -> None:
    """Load `name`'s slot tensors from `app.transformer.*`; every one must be stored."""
    keys = {
        f"{fqn}.lora.{name}.{key}"
        for fqn, pair in lora_pairs(transformer, name).items()
        for key in pair.state_dict()
    }
    stored = dcp.FileSystemReader(checkpoint).read_metadata().state_dict_metadata
    missing = sorted(key for key in keys if f"app.transformer.{key}" not in stored)
    if missing:
        raise ValueError(
            f"{checkpoint} lacks {len(missing)} tensors of variant {name!r}, "
            f"e.g. {missing[:3]}."
        )
    state = {
        key: value for key, value in transformer.state_dict().items() if key in keys
    }
    dcp.load(
        {"app": {"transformer": state}},
        checkpoint_id=checkpoint,
        no_dist=True,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    transformer.load_state_dict(state, strict=False)
    logger.info("Loaded variant %r from DCP checkpoint %s", name, checkpoint)


def _peft_copy(
    source: torch.nn.Module, name: str, lora: LoraVariant, target: Any
) -> None:
    """Give `target` a PEFT adapter with `name`'s slot layout holding the slot tensors."""
    pairs = lora_pairs(source, name)
    target.add_adapter(
        LoraConfig(
            r=lora.rank,
            # PEFT annotates an int, but any real alpha works: scaling = alpha / r.
            lora_alpha=lora.scaling * lora.rank,  # ty: ignore[invalid-argument-type]
            target_modules=sorted(pairs),
            lora_bias=lora.bias,
        )
    )
    for fqn, pair in pairs.items():
        layer = cast(LoraLayer, target.get_submodule(fqn))
        layer.lora_A["default"].load_state_dict(pair.A.state_dict(), assign=True)
        layer.lora_B["default"].load_state_dict(pair.B.state_dict(), assign=True)


def _load_external_lora(
    transformer: Any,
    lora_path: str,
    *,
    weight_name: str | None,
    adapter_name: str = "default",
) -> None:
    model_name = type(transformer).__name__
    try:
        loader_mixin = _DIFFUSERS_LORA_LOADERS[model_name]
    except KeyError:
        raise TypeError(
            f"No Diffusers pipeline LoRA converter is registered for {model_name}. "
            "The model adapter plugin must call "
            "register_diffusers_lora_loader() before using external LoRA formats."
        ) from None
    loader_api = cast(Any, loader_mixin)

    raw_state, metadata = _fetch_state_dict(
        pretrained_model_name_or_path_or_dict=lora_path,
        weight_name=weight_name,
        use_safetensors=True,
        local_files_only=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        token=None,
        revision=None,
        subfolder=None,
        user_agent={"file_type": "attn_procs_weights", "framework": "pytorch"},
        allow_pickle=False,
    )
    dora_keys = sorted(key for key in raw_state if "dora_scale" in key)
    if dora_keys:
        raise ValueError(
            "Diffusers would discard unsupported DoRA scale tensors; refusing a "
            f"lossy conversion: {dora_keys}"
        )
    kwargs: dict[str, Any] = {"return_lora_metadata": True}
    if loader_mixin is FluxLoraLoaderMixin:
        state_dict, network_alphas, _ = loader_api.lora_state_dict(
            raw_state,
            return_alphas=True,
            **kwargs,
        )
    else:
        state_dict, _ = loader_api.lora_state_dict(raw_state, **kwargs)
        network_alphas = None

    unsupported = [key for key in state_dict if not key.startswith("transformer.")]
    non_lora = [key for key in state_dict if "lora" not in key]
    if unsupported or non_lora:
        raise ValueError(
            "This converter handles transformer LoRA tensors only; the official "
            "Diffusers conversion produced unsupported keys: "
            f"{sorted(set(unsupported + non_lora))}"
        )

    load_kwargs = {
        "state_dict": state_dict,
        "transformer": transformer,
        "adapter_name": adapter_name,
        "metadata": metadata,
        "low_cpu_mem_usage": True,
    }
    if loader_mixin is FluxLoraLoaderMixin:
        load_kwargs["network_alphas"] = network_alphas
    loader_api.load_lora_into_transformer(**load_kwargs)
    if adapter_name not in getattr(transformer, "peft_config", {}):
        raise RuntimeError(
            f"Diffusers found no transformer LoRA weights at {lora_path!r}."
        )
    loaded_state = LoraBaseMixin.pack_weights(
        get_peft_model_state_dict(transformer, adapter_name=adapter_name),
        "transformer",
    )
    missing = sorted(set(state_dict) - set(loaded_state))
    unexpected = sorted(set(loaded_state) - set(state_dict))
    shape_mismatches = {
        key: (tuple(state_dict[key].shape), tuple(loaded_state[key].shape))
        for key in set(state_dict) & set(loaded_state)
        if state_dict[key].shape != loaded_state[key].shape
    }
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "Diffusers did not materialize every converted LoRA tensor; refusing "
            "a partial conversion. "
            f"not loaded: {missing}, not in input: {unexpected}, "
            f"shape mismatches: {shape_mismatches}"
        )


def _save_diffusers_lora(
    transformer: Any,
    output_dir: str,
    *,
    adapter_name: str = "default",
    weight_name: str = LORA_WEIGHT_NAME_SAFE,
) -> None:
    output = _require_new_output(output_dir)
    _validate_output_weight_name(weight_name)
    peft_config = getattr(transformer, "peft_config", {})
    if adapter_name not in peft_config:
        raise ValueError(f"Adapter {adapter_name!r} is not loaded in the transformer.")

    state_dict = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
    if not state_dict:
        raise RuntimeError(f"Adapter {adapter_name!r} has no LoRA tensors to save.")
    packed_state = LoraBaseMixin.pack_weights(state_dict, "transformer")
    packed_metadata = LoraBaseMixin.pack_weights(
        peft_config[adapter_name].to_dict(), "transformer"
    )
    output.mkdir(parents=True)
    metadata = {
        "format": "pt",
        LORA_ADAPTER_METADATA_KEY: json.dumps(
            packed_metadata,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
    }
    save_file(
        packed_state,
        output / weight_name,
        metadata=metadata,
    )
    logger.info("Saved Diffusers LoRA to %s", output / weight_name)


def _save_native_lora(
    transformer: Any, path: Path, adapter_name: str = "default"
) -> None:
    """Write a PEFT adapter as a native file (`<linear_fqn>.{A.weight,B.weight,B.bias}`)."""
    layers = {
        fqn: cast(LoraLayer, module)
        for fqn, module in transformer.named_modules()
        if isinstance(module, LoraLayer) and adapter_name in module.lora_A
    }
    shapes = {
        (layer.r[adapter_name], layer.scaling[adapter_name])
        for layer in layers.values()
    }
    if len(shapes) != 1:
        raise ValueError(
            f"A native LoRA file has one rank and alpha; the adapter has {sorted(shapes)}."
        )
    ((rank, scaling),) = shapes
    tensors = {
        f"{fqn}.{half}.{key}": tensor.detach().cpu().contiguous()
        for fqn, layer in layers.items()
        for half, module in (("A", layer.lora_A), ("B", layer.lora_B))
        for key, tensor in module[adapter_name].state_dict().items()
    }
    variant = LoraVariant(rank=rank, alpha=scaling * rank)
    path.parent.mkdir(parents=True)
    save_file(tensors, path, metadata=lora_metadata(variant))
    logger.info("Saved native LoRA to %s", path)


def _fuse_transformer(
    transformer: Any,
    output_dir: str,
    *,
    scale: float,
    adapter_name: str = "default",
) -> None:
    output = _require_new_output(output_dir)
    _validate_fuse_scale(scale)
    transformer.fuse_lora(
        lora_scale=scale,
        safe_fusing=True,
        adapter_names=[adapter_name],
    )
    transformer.unload_lora()
    transformer.save_pretrained(output, safe_serialization=True)
    logger.info("Saved fused Diffusers transformer to %s", output)


def export_dcp(
    config: dict[str, Any],
    *,
    checkpoint_dir: str,
    output_dir: str,
    variant: str = "default",
    weight_name: str = LORA_WEIGHT_NAME_SAFE,
) -> None:
    """Export one LoRA variant of a DCP training checkpoint as a Diffusers LoRA."""
    _require_new_output(output_dir)
    _validate_output_weight_name(weight_name)
    checkpoint = _require_dcp_checkpoint(checkpoint_dir)
    adapter = _make_model_adapter(config, variants=True)
    name, lora = _lora_slots(adapter, variant)
    _load_dcp_slots(adapter.transformer, checkpoint, name)
    # The PEFT host only carries the LoRA tensors: a fresh meta copy, not the cached one.
    HfModelLoader.invalidate_cache(adapter.hf_model.config_key)
    target = _make_model_adapter(config, variants=False, device="meta").transformer
    _peft_copy(adapter.transformer, name, lora, target)
    _save_diffusers_lora(target, output_dir, weight_name=weight_name)


def import_lora(
    config: dict[str, Any],
    *,
    lora_path: str,
    output_dir: str,
    name: str = "lora",
    weight_name: str | None = None,
) -> None:
    """Convert a Diffusers-compatible LoRA to a native file for a lora variant's ``path``."""
    output = _require_new_output(output_dir)
    adapter = _make_model_adapter(config, variants=False)
    _load_external_lora(adapter.transformer, lora_path, weight_name=weight_name)
    _save_native_lora(adapter.transformer, output / f"{name}.safetensors")


def convert(
    config: dict[str, Any],
    *,
    lora_path: str,
    output_dir: str,
    input_weight_name: str | None = None,
    output_weight_name: str = LORA_WEIGHT_NAME_SAFE,
) -> None:
    """Normalize a Diffusers-compatible input through its official loader."""
    _require_new_output(output_dir)
    _validate_output_weight_name(output_weight_name)
    adapter = _make_model_adapter(config, variants=False)
    _load_external_lora(
        adapter.transformer,
        lora_path,
        weight_name=input_weight_name,
    )
    _save_diffusers_lora(
        adapter.transformer,
        output_dir,
        weight_name=output_weight_name,
    )


def fuse(
    config: dict[str, Any],
    *,
    lora_path: str,
    output_dir: str,
    scale: float = 1.0,
    weight_name: str | None = None,
) -> None:
    """Fuse a Diffusers-compatible LoRA into a fresh CPU-loaded transformer."""
    _require_new_output(output_dir)
    _validate_fuse_scale(scale)
    adapter = _make_model_adapter(config, variants=False)
    transformer: Any = adapter.transformer
    _load_external_lora(
        transformer,
        lora_path,
        weight_name=weight_name,
    )
    _fuse_transformer(transformer, output_dir, scale=scale)
