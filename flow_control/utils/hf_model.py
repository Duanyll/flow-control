from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict

from .logging import get_logger
from .types import TorchDType

logger = get_logger(__name__)


class HfModelLoader(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    library: Literal["diffusers", "transformers"]
    class_name: str
    pretrained_model_id: str
    revision: str | None = None
    subfolder: str | None = None
    dtype: TorchDType | Literal["auto"] = "auto"

    extra_from_pretrained_kwargs: dict[str, Any] = {}

    _model: Any = None

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, value: Any):
        self._model = value

    def load_model(self, use_meta_device: bool = False) -> Any:
        if self.library == "diffusers":
            import diffusers

            model_cls = getattr(diffusers, self.class_name)
        elif self.library == "transformers":
            import transformers

            model_cls = getattr(transformers, self.class_name)
        else:
            raise ValueError(f"Unknown model library: {self.library}")

        if use_meta_device:
            with torch.device("meta"):
                config = model_cls.load_config(
                    self.pretrained_model_id,
                    revision=self.revision,
                    subfolder=self.subfolder,
                    **self.extra_from_pretrained_kwargs,
                )
                model = model_cls.from_config(config)
                if self.dtype != "auto":
                    model.to(dtype=self.dtype)
            logger.info(
                f"Initialized model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
                f"on meta device with dtype {self.dtype}"
            )
        else:
            if self.dtype != "auto":
                if self.library == "diffusers":
                    self.extra_from_pretrained_kwargs["torch_dtype"] = self.dtype
                elif self.library == "transformers":
                    self.extra_from_pretrained_kwargs["dtype"] = self.dtype
            model = model_cls.from_pretrained(
                self.pretrained_model_id,
                revision=self.revision,
                subfolder=self.subfolder,
                **self.extra_from_pretrained_kwargs,
            )
            logger.info(
                f"Loaded model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
                f"with dtype {self.dtype}"
            )
        self._model = model
        return model
