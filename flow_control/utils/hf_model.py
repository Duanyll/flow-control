from __future__ import annotations

import glob
import os
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Literal

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open

from . import device as devutil
from .logging import get_logger
from .types import TorchDType

logger = get_logger(__name__)

FP8_SCALE_SUFFIX = ".weight_scale"


def _hf_local_files_only() -> bool:
    """True when ``HF_HUB_OFFLINE`` is set, so loads resolve purely from the
    local cache.

    Passing ``local_files_only=True`` to ``from_pretrained`` / ``load_config``
    makes diffusers/transformers skip the ``model_info`` (``/api/models/...``)
    revision-resolution call entirely. That call is what blows up under
    ``HF_HUB_OFFLINE=1`` (the offline request hook *raises* instead of using the
    cache) and what depends on a flaky proxy when offline is unset. With a
    complete local cache this flag makes loading fully network-free.
    """
    return os.getenv("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


class LoadingScope:
    """Tracks ``config_key``s loaded during a model reload cycle.

    Created by :meth:`HfModelLoader.loading_scope`.  After the scope ends,
    call :meth:`purge_stale` to free cached models that were not loaded.
    """

    def __init__(self) -> None:
        self.active_keys: set[tuple] = set()

    def purge_stale(self) -> list[str]:
        """Remove cached models whose keys were not loaded in this scope."""
        stale = set(HfModelLoader._model_cache) - self.active_keys
        messages: list[str] = []
        for key in stale:
            del HfModelLoader._model_cache[key]
            # key layout: (library, class_name, pretrained_id, ...)
            messages.append(f"Purged cached model: {key[1]} ({key[2]})")
        if stale:
            devutil.empty_cache()
        return messages


class HfModelLoader[T](BaseModel):
    model_config = ConfigDict(extra="forbid")

    _model_cache: ClassVar[dict[tuple, Any]] = {}
    _scope_var: ClassVar[ContextVar[LoadingScope | None]] = ContextVar(
        "hf_model_loading_scope", default=None
    )

    @classmethod
    @contextmanager
    def loading_scope(cls) -> Generator[LoadingScope]:
        """Track which models are loaded; call ``scope.purge_stale()`` after."""
        scope = LoadingScope()
        token = cls._scope_var.set(scope)
        try:
            yield scope
        finally:
            cls._scope_var.reset(token)

    @classmethod
    def invalidate_cache(cls, key: tuple) -> None:
        """Remove a cache entry without touching any instance's ``_model``."""
        cls._model_cache.pop(key, None)

    library: Literal["diffusers", "transformers", "custom"]
    """Where to resolve ``class_name``. ``"custom"`` treats ``class_name`` as a
    dotted import path (e.g. ``"flow_control.adapters.x.modeling.XModel"``) to a
    transformers-style ``PreTrainedModel`` subclass (vendored/local modeling code)."""
    class_name: str
    pretrained_model_id: str
    revision: str = "main"
    subfolder: str | None = None
    dtype: TorchDType = torch.bfloat16
    device_memory_distribution: list[str] | None = None
    no_split_modules: list[str] | None = None
    dequantize: bool = True
    """Load a weight-only fp8 checkpoint (e4m3 ``<name>.weight`` beside a per-row
    ``<name>.weight_scale``) as plain ``dtype`` modules; False hands it to ``from_pretrained``."""

    extra_from_pretrained_kwargs: dict[str, Any] = {}

    _model: T | None = None

    @property
    def config_key(self) -> tuple:
        """Hashable key representing this loader's configuration.

        Two loaders with the same config_key will load the same model, so loaded
        weights can be transplanted between them.
        """
        return (
            self.library,
            self.class_name,
            self.pretrained_model_id,
            self.revision,
            self.subfolder or "",
            str(self.dtype),
            self.dequantize,
            tuple(self.device_memory_distribution or []),
            tuple(sorted(self.extra_from_pretrained_kwargs.items()))
            if self.extra_from_pretrained_kwargs
            else (),
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload_model(self) -> None:
        if self._model is None:
            return
        self._model_cache.pop(self.config_key, None)
        model = self._model
        self._model = None
        del model
        devutil.empty_cache()
        logger.info(
            f"Unloaded model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''}"
        )

    @property
    def model(self) -> T:
        if self._model is None:
            raise ValueError("Model not loaded yet. Call load_model() first.")
        return self._model

    @model.setter
    def model(self, value: T):
        self._model = value

    def _load_model_on_meta(self, model_cls, cast: bool = True) -> T:
        """Build from the config with parameters on meta; ``cast`` also moves the buffers to ``dtype``."""
        extra = dict(self.extra_from_pretrained_kwargs)
        extra.setdefault("local_files_only", _hf_local_files_only())
        with init_empty_weights():
            if self.library == "diffusers":
                config = model_cls.load_config(
                    self.pretrained_model_id,
                    revision=self.revision,
                    subfolder=self.subfolder,
                    **extra,
                )
                model = model_cls.from_config(config)
            else:
                # transformers requires subfolder to be a string ("" = root),
                # unlike diffusers' load_config which accepts None.
                config = model_cls.config_class.from_pretrained(
                    self.pretrained_model_id,
                    revision=self.revision,
                    subfolder=self.subfolder or "",
                    **extra,
                )
                model = model_cls(config)
            if cast:
                model.to(dtype=self.dtype)
        logger.info(
            f"Initialized model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
            f"on meta device with dtype {self.dtype}"
        )
        self._model = model
        return model

    def _load_with_from_pretrained(self, model_cls, device: torch.device) -> T:
        if self.dequantize and (files := self._fp8_files()):
            return self._load_dequantized(model_cls, device, files)
        kwargs = {
            **self.extra_from_pretrained_kwargs,
        }
        kwargs.setdefault("local_files_only", _hf_local_files_only())
        if self.revision != "main":
            kwargs["revision"] = self.revision
        if self.subfolder is not None:
            kwargs["subfolder"] = self.subfolder

        if self.library == "diffusers":
            kwargs["torch_dtype"] = self.dtype
        else:  # transformers and custom (transformers-style) classes
            kwargs["dtype"] = self.dtype
        model = model_cls.from_pretrained(
            self.pretrained_model_id,
            **kwargs,
        )
        logger.info(
            f"Loaded model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
            f"with dtype {self.dtype}"
        )
        if hasattr(model, "to"):
            model.to(device)
            logger.info(f"Moved model {self.class_name} to device {device}")
        self._model = model
        return model

    # ------------------------- weight-only fp8 checkpoints ---------------------- #

    def _fp8_files(self) -> list[str]:
        """The checkpoint's safetensors files when they hold ``weight_scale`` tensors."""
        pattern = os.path.join(self.subfolder or "", "*.safetensors")
        root = self.pretrained_model_id
        if not os.path.isdir(root):
            root = snapshot_download(
                root,
                revision=self.revision,
                allow_patterns=[pattern],
                ignore_patterns=None if self.subfolder else ["*/*"],
                local_files_only=_hf_local_files_only(),
            )
        files = sorted(glob.glob(os.path.join(root, pattern)))
        for file in files:
            with safe_open(file, "pt", "cpu") as reader:
                if any(key.endswith(FP8_SCALE_SUFFIX) for key in reader.keys()):  # noqa: SIM118
                    return files
        return []

    def _convert(
        self, key: str, tensor: torch.Tensor
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Checkpoint entry -> state-dict entries on the dequantizing path; subclasses map key layouts."""
        yield key, tensor

    def _load_dequantized(self, model_cls, device: torch.device, files: list[str]) -> T:
        """Build from the config and assign the state dict, dequantizing each fp8 weight.

        ``w.to(dtype) * scale.to(dtype)[:, None]`` is Ideogram 4's official
        ``Fp8Linear.forward``, so the numerics match it. Every floating tensor is
        cast to ``dtype`` (no fp32 modules are kept) and buffers computed by
        ``__init__`` (RoPE tables) stay as built.
        """
        model: Any = self._load_model_on_meta(model_cls, cast=False)
        state: dict[str, torch.Tensor] = {}
        with ExitStack() as stack:
            readers = [
                stack.enter_context(safe_open(f, "pt", str(device))) for f in files
            ]
            # safetensors readers expose keys() but are not iterable.
            where = {key: reader for reader in readers for key in reader.keys()}  # noqa: SIM118
            for key, reader in where.items():
                if key.endswith(FP8_SCALE_SUFFIX):
                    continue
                tensor = reader.get_tensor(key)
                if (scale_key := key + "_scale") in where:
                    scale = where[scale_key].get_tensor(scale_key)
                    if (
                        tensor.dtype != torch.float8_e4m3fn
                        or scale.shape != tensor.shape[:1]
                    ):
                        raise ValueError(
                            f"{self.pretrained_model_id}/{self.subfolder or ''}: {key} is "
                            f"{tensor.dtype} with a {tuple(scale.shape)} scale; only e4m3 "
                            "weights with one scale per row are dequantized (dequantize=false "
                            "hands the checkpoint to from_pretrained)."
                        )
                    tensor = tensor.to(self.dtype) * scale.to(self.dtype)[:, None]
                elif tensor.is_floating_point():
                    tensor = tensor.to(self.dtype)
                state.update(self._convert(key, tensor))
        model.load_state_dict(state, assign=True)
        model.to(device).eval()
        logger.info(
            f"Loaded fp8 weight-only {self.class_name} from {self.pretrained_model_id}/"
            f"{self.subfolder or ''} dequantized to {self.dtype} on {device}"
        )
        return model

    def _load_with_device_map(self, model_cls, device: torch.device) -> T:
        assert self.device_memory_distribution is not None
        if device.type != "cuda":
            raise ValueError(
                "Device memory distribution is only supported for CUDA devices."
            )

        # Load the model on meta to infer the device map based on the specified memory distribution
        model: Any = self._load_model_on_meta(model_cls)
        no_split_modules = self.no_split_modules or model._no_split_modules
        if not no_split_modules:
            logger.warning(
                f"Model {self.class_name} does not specify _no_split_modules, which may lead to suboptimal device mapping or raise tensor not on device errors."
            )
        device_idx = device.index if device.index is not None else 0
        device_map = infer_auto_device_map(
            model,
            max_memory={
                (device_idx + i): mem
                for i, mem in enumerate(self.device_memory_distribution)
            },
            no_split_module_classes=no_split_modules,
        )
        device_str = ", ".join(
            f"cuda:{device_idx + i}={mem}"
            for i, mem in enumerate(self.device_memory_distribution)
        )
        logger.info(f"Inferred device map for {self.class_name} with {device_str}")
        logger.debug(f"Inferred device map for {self.class_name}: {device_map}")

        # Call from_pretrained again with the inferred device map
        kwargs = {
            "revision": self.revision,
            "subfolder": self.subfolder,
            "device_map": device_map,
            **self.extra_from_pretrained_kwargs,
        }

        if self.library == "diffusers":
            kwargs["torch_dtype"] = self.dtype
        else:  # transformers and custom (transformers-style) classes
            kwargs["dtype"] = self.dtype
        model = model_cls.from_pretrained(
            self.pretrained_model_id,
            **kwargs,
        )
        logger.info(
            f"Loaded model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
            f"with dtype {self.dtype} and device map {device_str}"
        )
        self._model = model
        return model

    def _resolve_model_class(self) -> Any:
        if self.library == "diffusers":
            import diffusers

            return getattr(diffusers, self.class_name)
        if self.library == "transformers":
            import transformers

            return getattr(transformers, self.class_name)
        if self.library == "custom":
            import importlib

            module_path, _, cls_name = self.class_name.rpartition(".")
            if not module_path:
                raise ValueError(
                    f"library='custom' requires a dotted import path in class_name, "
                    f"got {self.class_name!r}"
                )
            return getattr(importlib.import_module(module_path), cls_name)
        raise ValueError(f"Unknown model library: {self.library}")

    def load_model(self, device: torch.device, frozen: bool = True) -> bool:
        """Load the model onto *device*.

        Returns ``True`` if a fresh load from pretrained was performed (caller
        should run post-load setup).  Returns ``False`` when the model was
        already loaded on this instance or was reused from the class-level
        cache.
        """
        key = self.config_key
        scope = self._scope_var.get(None)
        if scope is not None:
            scope.active_keys.add(key)

        # Already loaded on this instance
        if self._model is not None:
            return False

        # Reuse from cache
        if key in self._model_cache:
            self._model = self._model_cache[key]
            logger.info(
                f"Reusing cached {self.class_name} ({self.pretrained_model_id})"
            )
            if hasattr(self._model, "requires_grad_"):
                self._model.requires_grad_(not frozen)
            return False

        # Fresh load from pretrained
        model_cls = self._resolve_model_class()

        if device.type == "meta":
            self._load_model_on_meta(model_cls)
        elif self.device_memory_distribution is not None:
            self._load_with_device_map(model_cls, device)
        else:
            self._load_with_from_pretrained(model_cls, device)

        requires_grad_ = getattr(self.model, "requires_grad_", None)
        if requires_grad_ is not None:
            requires_grad_(not frozen)

        self._model_cache[key] = self._model
        return True
