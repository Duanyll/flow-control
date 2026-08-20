"""Offline LoRA conversion and fusion backed by Diffusers and PEFT."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

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
from peft.tuners.lora.layer import LoraLayer
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from flow_control.adapters import parse_model_adapter
from flow_control.adapters.base import BaseModelAdapter
from flow_control.training.ema import EMAConfig, EMAOptimizer
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

CheckpointWeights = Literal["current", "ema", "ema_old"]

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
    config: dict[str, Any],
    *,
    include_training_adapter: bool,
) -> BaseModelAdapter:
    raw_model = config.get("model")
    if not isinstance(raw_model, dict):
        raise ValueError("The config must contain a model object.")

    model_config = copy.deepcopy(raw_model)
    hf_model = model_config.get("hf_model")
    if not isinstance(hf_model, dict) or hf_model.get("library") != "diffusers":
        raise ValueError(
            "The first LoRA CLI implementation supports Diffusers transformers "
            "only (model.hf_model.library must be 'diffusers')."
        )
    # A CPU-only converter must not inherit a serving/training CUDA device map.
    hf_model["device_memory_distribution"] = None

    if not include_training_adapter:
        model_config["peft_lora_rank"] = 0
        model_config["all_trainable"] = False
        model_config["extra_trainable_modules"] = []

    adapter = parse_model_adapter(model_config)
    if include_training_adapter:
        _validate_training_lora(adapter)
    adapter.load_transformer(torch.device("cpu"))
    if not hasattr(adapter.transformer, "load_lora_adapter"):
        raise TypeError(
            f"{type(adapter.transformer).__name__} does not expose Diffusers "
            "load_lora_adapter()."
        )
    if include_training_adapter:
        _validate_pure_lora_parameters(adapter.transformer)
    return adapter


def _validate_training_lora(adapter: BaseModelAdapter) -> None:
    if adapter.peft_lora_rank <= 0:
        raise ValueError(
            "DCP LoRA export requires model.peft_lora_rank > 0 so the adapter "
            "state has a concrete PEFT skeleton."
        )
    if adapter.all_trainable or adapter.extra_trainable_modules:
        raise ValueError(
            "DCP LoRA export does not support full fine-tuning or "
            "extra_trainable_modules because Diffusers LoRA output cannot "
            "represent those parameters."
        )


def _validate_pure_lora_parameters(
    transformer: torch.nn.Module,
    adapter_name: str = "default",
    *,
    require_trainable: bool = True,
) -> None:
    trainable = [
        name
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable
        if f".{adapter_name}." not in name or "lora_" not in name
    ]
    if require_trainable and not trainable:
        raise RuntimeError(
            f"LoRA adapter {adapter_name!r} has no trainable parameters."
        )
    if unexpected:
        raise ValueError(
            "Offline LoRA conversion only represents PEFT LoRA tensors, but the "
            f"model also has trainable parameters: {unexpected}"
        )


class _DcpLoraState:
    """Training-checkpoint-shaped stateful that loads transformer and EMA only."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        checkpoint_weights: CheckpointWeights,
    ) -> None:
        self.transformer = transformer
        self.checkpoint_weights = checkpoint_weights
        _validate_pure_lora_parameters(transformer)
        trainable = [param for param in transformer.parameters() if param.requires_grad]
        if not trainable:
            raise RuntimeError("The transformer has no trainable LoRA parameters.")
        self.ema_optimizer = (
            EMAOptimizer(trainable, EMAConfig())
            if checkpoint_weights != "current"
            else None
        )

    @property
    def ema_key(self) -> str:
        return "optim_ema" if self.checkpoint_weights == "ema" else "optim_ema_old"

    def state_dict(self) -> dict[str, Any]:
        options = StateDictOptions(strict=False, ignore_frozen_params=True)
        state: dict[str, Any] = {
            "transformer": get_model_state_dict(self.transformer, options=options)
        }
        if self.ema_optimizer is not None:
            state[self.ema_key] = get_optimizer_state_dict(
                self.transformer,
                self.ema_optimizer,
                options=options,
            )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        options = StateDictOptions(strict=False, ignore_frozen_params=True)
        set_model_state_dict(
            self.transformer,
            state_dict["transformer"],
            options=options,
        )
        if self.ema_optimizer is not None:
            set_optimizer_state_dict(
                self.transformer,
                self.ema_optimizer,
                state_dict[self.ema_key],
                options=options,
            )
            self.ema_optimizer.coerce_buffer_dtype()

    def apply_selected_weights(self) -> None:
        if self.ema_optimizer is not None:
            self.ema_optimizer.apply_shadow()


def _validate_dcp_payload(
    checkpoint: Path,
    checkpoint_weights: CheckpointWeights,
) -> None:
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    keys = metadata.state_dict_metadata.keys()
    if not any(key.startswith("app.transformer.") for key in keys):
        raise ValueError(f"DCP checkpoint has no app.transformer state: {checkpoint}")
    if checkpoint_weights != "current":
        ema_key = "optim_ema" if checkpoint_weights == "ema" else "optim_ema_old"
        if not any(key.startswith(f"app.{ema_key}.") for key in keys):
            raise ValueError(
                f"DCP checkpoint has no app.{ema_key} state required by "
                f"--weights {checkpoint_weights}: {checkpoint}"
            )


def _load_dcp_lora(
    transformer: torch.nn.Module,
    checkpoint_dir: str,
    checkpoint_weights: CheckpointWeights,
) -> None:
    checkpoint = _require_dcp_checkpoint(checkpoint_dir)
    _validate_dcp_payload(checkpoint, checkpoint_weights)
    state = _DcpLoraState(transformer, checkpoint_weights)
    dcp.load(
        {"app": state},
        checkpoint_id=checkpoint,
        no_dist=True,
        # Extra state from a full training checkpoint is fine, but every LoRA
        # tensor in our configured skeleton must be present.
        planner=DefaultLoadPlanner(allow_partial_load=False),
    )
    state.apply_selected_weights()
    logger.info(
        "Loaded %s LoRA weights from DCP checkpoint %s",
        checkpoint_weights,
        checkpoint,
    )


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


@torch.no_grad()
def _copy_lora_adapter(
    transformer: Any,
    *,
    source_name: str,
    target_name: str,
) -> None:
    """Copy between adapters only when their effective PEFT layouts agree."""
    source_state = get_peft_model_state_dict(
        transformer,
        adapter_name=source_name,
    )
    target_state = get_peft_model_state_dict(
        transformer,
        adapter_name=target_name,
    )
    _validate_lora_state_layout(source_state, target_state)
    semantic_mismatches = _lora_semantic_mismatches(
        transformer,
        source_name=source_name,
        target_name=target_name,
    )
    if semantic_mismatches:
        raise ValueError(
            "The imported LoRA scaling/configuration is incompatible with the "
            "configured training adapter: " + ", ".join(semantic_mismatches)
        )

    incompatible = set_peft_model_state_dict(
        transformer,
        source_state,
        adapter_name=target_name,
    )
    if incompatible.unexpected_keys:
        raise ValueError(
            "The configured training adapter cannot represent these imported "
            f"LoRA tensors: {sorted(incompatible.unexpected_keys)}"
        )
    _rescale_target_adapter(
        transformer,
        source_name=source_name,
        target_name=target_name,
    )

    transformer.delete_adapters(source_name)
    transformer.set_adapters(target_name)


def _validate_lora_state_layout(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> None:
    if not source_state or not target_state:
        raise RuntimeError("Source and target LoRA adapters must both have parameters.")

    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        raise ValueError(
            "The imported and configured LoRA target layouts differ. Configure "
            "the same target modules and adapter variant before importing; "
            f"missing from input: {sorted(target_keys - source_keys)}, "
            f"not representable by target: {sorted(source_keys - target_keys)}"
        )
    shape_mismatches = {
        key: (tuple(source_state[key].shape), tuple(target_state[key].shape))
        for key in source_keys
        if source_state[key].shape != target_state[key].shape
    }
    if shape_mismatches:
        raise ValueError(
            f"The imported and configured LoRA ranks/shapes differ: {shape_mismatches}"
        )


def _lora_semantic_mismatches(
    transformer: torch.nn.Module,
    *,
    source_name: str,
    target_name: str,
) -> list[str]:
    semantic_mismatches: list[str] = []
    for module_name, module in transformer.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        has_source = source_name in module.r
        has_target = target_name in module.r
        if has_source != has_target:
            semantic_mismatches.append(f"{module_name}: adapter coverage")
            continue
        if not has_source:
            continue
        semantic_mismatches.extend(
            f"{module_name}: {reason}"
            for reason in _lora_layer_mismatches(module, source_name, target_name)
        )
    return semantic_mismatches


def _lora_layer_mismatches(
    module: LoraLayer,
    source_name: str,
    target_name: str,
) -> list[str]:
    mismatches = []
    for attribute in ("r", "use_dora", "lora_bias"):
        values = getattr(module, attribute)
        if values[source_name] != values[target_name]:
            mismatches.append(attribute)

    if type(module.lora_variant.get(source_name)) is not type(
        module.lora_variant.get(target_name)
    ):
        mismatches.append("LoRA variant")

    source_scale = float(module.scaling[source_name])
    target_scale = float(module.scaling[target_name])
    if (
        not math.isfinite(source_scale)
        or not math.isfinite(target_scale)
        or target_scale == 0.0
    ):
        mismatches.append("invalid scaling")
    elif source_scale != target_scale and (
        module.use_dora[source_name]
        or source_name in module.lora_variant
        or target_name in module.lora_variant
    ):
        mismatches.append("variant scaling cannot be reparameterized")
    return mismatches


def _rescale_target_adapter(
    transformer: torch.nn.Module,
    *,
    source_name: str,
    target_name: str,
) -> None:
    """Preserve the effective delta when source and target alpha differ."""
    for module_name, module in transformer.named_modules():
        if not isinstance(module, LoraLayer) or source_name not in module.r:
            continue
        ratio = float(module.scaling[source_name]) / float(module.scaling[target_name])
        if ratio == 1.0:
            continue
        if target_name in module.lora_B:
            target_b = module.lora_B[target_name]
            cast(torch.Tensor, target_b.weight).mul_(ratio)
            target_bias = cast(torch.Tensor | None, target_b.bias)
            if target_bias is not None:
                target_bias.mul_(ratio)
        elif target_name in module.lora_embedding_B:
            module.lora_embedding_B[target_name].mul_(ratio)
        else:
            raise ValueError(
                f"Cannot reparameterize LoRA scaling for module {module_name!r}."
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
    _validate_pure_lora_parameters(
        transformer,
        adapter_name,
        require_trainable=False,
    )

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


def _save_dcp_adapter(
    adapter: BaseModelAdapter,
    output_dir: str,
) -> None:
    output = _require_new_output(output_dir)
    _validate_pure_lora_parameters(adapter.transformer)
    options = StateDictOptions(strict=False, ignore_frozen_params=True)
    transformer_state = get_model_state_dict(adapter.transformer, options=options)
    if not transformer_state or not any("lora_" in key for key in transformer_state):
        raise RuntimeError("No trainable LoRA parameters were found for DCP output.")

    dcp.save(
        {"app": {"transformer": transformer_state}},
        checkpoint_id=output,
        no_dist=True,
    )
    logger.info("Saved transformer-only DCP LoRA checkpoint to %s", output)


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
    checkpoint_weights: CheckpointWeights = "current",
    adapter_name: str = "default",
    weight_name: str = LORA_WEIGHT_NAME_SAFE,
) -> None:
    """Export one weight view from a DCP training checkpoint to Diffusers LoRA."""
    _require_new_output(output_dir)
    _validate_output_weight_name(weight_name)
    checkpoint = _require_dcp_checkpoint(checkpoint_dir)
    _validate_dcp_payload(checkpoint, checkpoint_weights)
    adapter = _make_model_adapter(config, include_training_adapter=True)
    _load_dcp_lora(
        adapter.transformer,
        checkpoint_dir,
        checkpoint_weights,
    )
    _save_diffusers_lora(
        adapter.transformer,
        output_dir,
        adapter_name=adapter_name,
        weight_name=weight_name,
    )


def import_dcp(
    config: dict[str, Any],
    *,
    lora_path: str,
    output_dir: str,
    weight_name: str | None = None,
) -> None:
    """Convert a Diffusers-compatible LoRA to transformer-only DCP state."""
    _require_new_output(output_dir)
    adapter = _make_model_adapter(config, include_training_adapter=True)
    _load_external_lora(
        adapter.transformer,
        lora_path,
        weight_name=weight_name,
        adapter_name="imported",
    )
    _copy_lora_adapter(
        adapter.transformer,
        source_name="imported",
        target_name="default",
    )
    _save_dcp_adapter(adapter, output_dir)


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
    adapter = _make_model_adapter(config, include_training_adapter=False)
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
    adapter = _make_model_adapter(config, include_training_adapter=False)
    transformer: Any = adapter.transformer
    _load_external_lora(
        transformer,
        lora_path,
        weight_name=weight_name,
    )
    _fuse_transformer(transformer, output_dir, scale=scale)
