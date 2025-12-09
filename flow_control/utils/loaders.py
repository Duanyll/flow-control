import json
import tomllib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from .logging import get_logger
from .types import TorchDType

logger = get_logger(__name__)


class HfModelLoader(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["diffusers", "transformers"]
    class_name: str
    pretrained_model_id: str
    revision: str | None = None
    subfolder: str | None = None
    dtype: TorchDType | Literal["auto"] = "auto"

    def load_model(self) -> Any:
        if self.type == "diffusers":
            import diffusers

            model_cls = getattr(diffusers, self.class_name)
        elif self.type == "transformers":
            import transformers

            model_cls = getattr(transformers, self.class_name)
        else:
            raise ValueError(f"Unknown model type: {self.type}")

        model = model_cls.from_pretrained(
            self.pretrained_model_id,
            revision=self.revision,
            subfolder=self.subfolder,
            torch_dtype=None if self.dtype == "auto" else self.dtype,
        )
        logger.info(
            f"Loaded model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} "
            f"with dtype {self.dtype}"
        )
        return model


def load_config_file(path: str) -> dict:
    """
    Load JSON, YAML or TOML configuration file.
    """
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    elif path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif path.endswith(".toml"):
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path}")
