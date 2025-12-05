from typing import Any, Literal

from pydantic import BaseModel

from .types import TorchDType
from .logging import get_logger

logger = get_logger(__name__)

class HfModelLoader(BaseModel):
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
        logger.info(f"Loaded model {self.class_name} from {self.pretrained_model_id}/{self.subfolder or ''} with dtype {self.dtype}")
        return model