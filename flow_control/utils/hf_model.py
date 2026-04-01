from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Literal

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from pydantic import BaseModel, ConfigDict

from .logging import get_logger
from .types import TorchDType

logger = get_logger(__name__)


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
        if stale and torch.cuda.is_available():
            torch.cuda.empty_cache()
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

    library: Literal["diffusers", "transformers"]
    class_name: str
    pretrained_model_id: str
    revision: str = "main"
    subfolder: str | None = None
    dtype: TorchDType = torch.bfloat16
    device_memory_distribution: list[str] | None = None
    no_split_modules: list[str] | None = None

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

    def _load_model_on_meta(self, model_cls) -> T:
        with init_empty_weights():
            if self.library == "diffusers":
                config = model_cls.load_config(
                    self.pretrained_model_id,
                    revision=self.revision,
                    subfolder=self.subfolder,
                    **self.extra_from_pretrained_kwargs,
                )
                model = model_cls.from_config(config)
            else:
                config = model_cls.config_class.from_pretrained(
                    self.pretrained_model_id,
                    revision=self.revision,
                    subfolder=self.subfolder,
                    **self.extra_from_pretrained_kwargs,
                )
                model = model_cls(config)
            model.to(dtype=self.dtype)
        logger.info(
            f"Initialized model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
            f"on meta device with dtype {self.dtype}"
        )
        self._model = model
        return model

    def _load_with_from_pretrained(self, model_cls, device: torch.device) -> T:
        kwargs = {
            **self.extra_from_pretrained_kwargs,
        }
        if self.revision != "main":
            kwargs["revision"] = self.revision
        if self.subfolder is not None:
            kwargs["subfolder"] = self.subfolder

        if self.library == "diffusers":
            kwargs["torch_dtype"] = self.dtype
        elif self.library == "transformers":
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
        elif self.library == "transformers":
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
                self._model.requires_grad_(not frozen)  # type: ignore
            return False

        # Fresh load from pretrained
        if self.library == "diffusers":
            import diffusers

            model_cls = getattr(diffusers, self.class_name)
        elif self.library == "transformers":
            import transformers

            model_cls = getattr(transformers, self.class_name)
        else:
            raise ValueError(f"Unknown model library: {self.library}")

        if device.type == "meta":
            self._load_model_on_meta(model_cls)
        elif self.device_memory_distribution is not None:
            self._load_with_device_map(model_cls, device)
        else:
            self._load_with_from_pretrained(model_cls, device)

        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(not frozen)  # type: ignore

        self._model_cache[key] = self._model
        return True
