from typing import Literal, Any
from pydantic import BaseModel
from .types import TorchDType

class HfModelLoader(BaseModel):
    type: Literal["diffusers", "transformers", "timm"]
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
        elif self.type == "timm":
            import timm
            model_cls = getattr(timm, self.class_name)
        else:
            raise ValueError(f"Unknown model type: {self.type}")
        
        model = model_cls.from_pretrained(
            self.pretrained_model_id,
            revision=self.revision,
            subfolder=self.subfolder,
            torch_dtype=None if self.dtype == "auto" else self.dtype,
        )
        return model