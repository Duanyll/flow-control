from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import ConfigDict

from flow_control.utils.remote import RemoteOffloadable


class BaseReward(RemoteOffloadable, ABC):
    """Base class for reward functions.

    The ``score`` method receives a single-sample batch dict that contains all
    information the trainer has about the sample (prompt text, generated image,
    model embeddings, etc.).  Each reward implementation extracts whatever
    fields it needs from the dict.

    Subclasses implement ``_load_model``, ``_score``, and optionally
    ``_unload_model``.  The public ``load_model`` / ``score`` / ``unload_model``
    methods handle transparent remote offloading when ``endpoint`` is set.
    """

    type: str
    weight: float = 1.0
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    @abstractmethod
    def _batch_fields(self) -> set[str]:
        """Set of expected batch keys for this reward."""
        ...

    @abstractmethod
    def _load_model(self, device: torch.device) -> None:
        """Load reward model to specified device. Called once before training."""
        ...

    @abstractmethod
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
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

    def _unload_model(self) -> None:
        """Optional: unload reward model to free GPU memory."""

    # ── Public API (handles remote dispatch transparently) ───────────────

    def load_model(self, device: torch.device) -> None:
        """Load reward model. If ``endpoint`` is set, offloads to remote server."""
        if self.endpoint is not None:
            self._init_remote(device)
        else:
            self._load_model(device)

    def score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute reward score. If remote, filters batch by ``_batch_fields``
        and sends only the needed keys to save bandwidth."""
        if self.is_remote:
            return self._remote_batch_call("/score", batch, fields=self._batch_fields)
        return self._score(batch)

    async def async_score(self, batch: dict[str, Any]) -> torch.Tensor:
        """Async version of ``score`` for use in async RL loops."""
        if self.is_remote:
            return await self._async_remote_batch_call(
                "/score", batch, fields=self._batch_fields
            )
        return self._score(batch)

    def unload_model(self) -> None:
        """Unload reward model (local or remote)."""
        if self.is_remote:
            self._close_remote()
        else:
            self._unload_model()
