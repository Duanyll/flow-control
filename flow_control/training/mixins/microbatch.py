"""Shared batching configuration and arithmetic for the training loops.

Naming convention: ``train_batch_size`` counts logical samples per optimizer
update *globally* (world-size-invariant, divided across ranks), while
``train_micro_batch_size`` is a per-rank physical forward size — a pure
throughput/memory knob that never changes training dynamics. Changing
world_size must never silently change dynamics; violated divisibility raises.

SFT uses the fields and validation properties (its update loop is
DataLoader-driven). The list-driven RL trainers (GRPO/NFT/AWM/RAM) also share
the chunk (one optimizer update) / microbatch (one model forward) slicing via
``iter_micro_updates``; the side effects (gradient-sync flag, backward,
optimizer step) stay in each trainer's visible loop.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from pydantic import BaseModel, PositiveInt

from flow_control.utils.logging import get_logger, warn_once

from .base import BaseTrainer

logger = get_logger(__name__)


@dataclass(slots=True)
class MicroUpdate[T]:
    items: list[T]
    loss_scale: float
    """``len(items) / chunk_size``; sums to 1 over each optimizer chunk, so a
    trainer computing a microbatch-mean loss backwards ``loss * loss_scale``."""
    is_sync_step: bool
    """Last microbatch of its optimizer chunk: enable gradient sync before the
    backward and run the optimizer step after it."""


class RolloutIndexedItem(Protocol):
    rollout_idx: int


class MicrobatchTrainMixin(BaseTrainer, BaseModel):
    train_batch_size: int = 4
    """
    How many (rollout, timestep) items the optimizer sees per update step across
    all ranks. Must be divisible by world_size.
    """
    train_micro_batch_size: PositiveInt = 1
    """Number of logical items combined in each per-rank model forward."""

    @property
    def local_train_batch_size(self) -> int:
        if self.train_batch_size % self.world_size != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible "
                f"by world_size ({self.world_size})."
            )
        return self.train_batch_size // self.world_size

    @property
    def grad_acc_steps(self) -> int:
        if self.local_train_batch_size % self.train_micro_batch_size != 0:
            raise ValueError(
                f"Per-rank train batch ({self.local_train_batch_size}) must be "
                f"divisible by train_micro_batch_size "
                f"({self.train_micro_batch_size})."
            )
        return self.local_train_batch_size // self.train_micro_batch_size

    def iter_train_micro_batches[T](self, items: Sequence[T]) -> Iterator[list[T]]:
        """Slice ``items`` into ``train_micro_batch_size`` groups, keeping order."""
        for start in range(0, len(items), self.train_micro_batch_size):
            yield list(items[start : start + self.train_micro_batch_size])

    def iter_micro_updates[T](
        self, train_items: Sequence[T]
    ) -> Iterator[MicroUpdate[T]]:
        """Slice one inner epoch's train items into microbatch updates.

        Consumers must run one backward per yielded update, scaled by
        ``loss_scale``, and step the optimizer on ``is_sync_step``. All ranks
        must submit the same number of items so their per-microbatch model
        collectives line up.
        """
        if not train_items:
            raise RuntimeError(
                f"{type(self).__name__} received no train items for this inner epoch."
            )
        self._warn_if_tail_update(len(train_items))
        for chunk_start in range(0, len(train_items), self.local_train_batch_size):
            chunk = train_items[chunk_start : chunk_start + self.local_train_batch_size]
            micro_batches = list(self.iter_train_micro_batches(chunk))
            for index, micro_items in enumerate(micro_batches):
                yield MicroUpdate(
                    items=micro_items,
                    loss_scale=len(micro_items) / len(chunk),
                    is_sync_step=index == len(micro_batches) - 1,
                )

    def _warn_if_tail_update(self, num_train_items: int) -> None:
        if num_train_items % self.local_train_batch_size != 0 and self.is_main_process:
            warn_once(
                logger,
                f"Local loss count ({num_train_items}) is not divisible by the "
                f"per-rank train batch ({self.local_train_batch_size}). "
                "The tail update uses a smaller effective batch.",
            )

    def _check_finite_loss(
        self, loss: torch.Tensor, items: Sequence[RolloutIndexedItem]
    ) -> None:
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite {type(self).__name__} loss detected: {loss.item()}. "
                f"(rollout_indices={[item.rollout_idx for item in items]})"
            )
