from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class BaseReward(BaseModel, ABC):
    """Base class for reward functions.

    The ``score`` method receives a single-sample batch dict that contains all
    information the trainer has about the sample (prompt text, generated image,
    model embeddings, etc.).  Each reward implementation extracts whatever
    fields it needs from the dict.
    """

    type: str
    weight: float = 1.0
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def load_model(self, device: torch.device) -> None:
        """Load reward model to specified device. Called once before training."""
        ...

    @abstractmethod
    def score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute reward score for a single sample.

        Args:
            batch: dict containing sample information. Common keys include:
                - ``clean_image``: [1, C, H, W] tensor in [0, 1] range
                - ``prompt``: str, the original prompt text
                Other keys depend on the processor / task.

        Returns:
            Scalar float tensor (shape ``[]`` or ``[1]``).
        """
        ...

    def unload_model(self) -> None:
        """Optional: unload reward model to free GPU memory."""
