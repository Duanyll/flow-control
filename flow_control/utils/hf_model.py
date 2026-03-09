from typing import Any, Literal

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from pydantic import BaseModel, ConfigDict

from .logging import get_logger
from .types import TorchDType

logger = get_logger(__name__)


class HfModelLoader[T](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    library: Literal["diffusers", "transformers"]
    class_name: str
    pretrained_model_id: str
    revision: str = "main"
    subfolder: str | None = None
    dtype: TorchDType | Literal["auto"] = "auto"
    device_memory_distribution: list[str] | None = None
    no_split_modules: list[str] | None = None

    extra_from_pretrained_kwargs: dict[str, Any] = {}

    _model: T | None = None

    # @classmethod
    # def __get_pydantic_json_schema__(
    #     cls, source: Any, handler: GetJsonSchemaHandler
    # ) -> JsonSchemaValue:
    #     # Generic type parameter T is not JSON-serializable; emit a fixed
    #     # schema based on the declared fields (excluding T).
    #     return {
    #         "type": "object",
    #         "properties": {
    #             "library": {"enum": ["diffusers", "transformers"]},
    #             "class_name": {"type": "string"},
    #             "pretrained_model_id": {"type": "string"},
    #             "revision": {"type": "string", "default": "main"},
    #             "subfolder": {"type": ["string", "null"], "default": None},
    #             "dtype": {"type": "string", "default": "auto"},
    #             "device_memory_distribution": {
    #                 "type": ["array", "null"],
    #                 "items": {"type": "string"},
    #                 "default": None,
    #             },
    #             "no_split_modules": {
    #                 "type": ["array", "null"],
    #                 "items": {"type": "string"},
    #                 "default": None,
    #             },
    #             "extra_from_pretrained_kwargs": {
    #                 "type": "object",
    #                 "additionalProperties": True,
    #                 "default": {},
    #             },
    #         },
    #         "required": ["library", "class_name", "pretrained_model_id"],
    #     }

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
            if self.dtype != "auto":
                model.to(dtype=self.dtype)
        logger.info(
            f"Initialized model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
            f"on meta device with dtype {self.dtype}"
        )
        self._model = model
        return model

    def _load_with_from_pretrained(self, model_cls, device: torch.device) -> T:
        kwargs = {
            "revision": self.revision,
            "subfolder": self.subfolder,
            **self.extra_from_pretrained_kwargs,
        }
        if self.dtype != "auto":
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
        if self.dtype != "auto":
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

    def load_model(self, device: torch.device) -> T:
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

        return self.model
