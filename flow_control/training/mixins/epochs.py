"""``EpochLoopMixin``: the step-counted update loop over a ``RowStream`` (SFT, VAE).

An epoch is one pass over the rank's planned rows, cut into optimizer updates of
``grad_acc_steps`` microbatches. The stream drops the tail short of one update at
plan level (design §7.2: ``len(loader)`` is known, the epoch arithmetic follows),
so the update boundary depends on the position inside the epoch alone and the only
data cursor is the ``StatefulDataLoader``'s yielded count, which the trainer
checkpoints. Checkpoints are taken after a sync step only, so a resumed iterator
starts on an update boundary.
"""

from abc import abstractmethod
from collections.abc import Iterator

from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.data import Planner, Row, RowStore, RowStream, build_loader

from .base import BaseTrainer


class EpochLoopMixin(BaseTrainer):
    train_steps: int
    """Optimizer updates to run in total."""

    _stream: RowStream
    _dataloader: StatefulDataLoader
    _current_step: int = 0

    @property
    @abstractmethod
    def grad_acc_steps(self) -> int:
        """Microbatches per optimizer update, provided by the trainer. A base that
        provides it (``MicrobatchTrainMixin``) must precede this mixin in the
        class bases so its concrete property wins in the MRO."""

    @property
    def steps_per_epoch(self) -> int:
        """Optimizer updates per epoch (exact: the stream holds whole updates)."""
        return len(self._dataloader) // self.grad_acc_steps

    @property
    def current_epoch(self) -> int:
        return self._current_step // self.steps_per_epoch

    def make_train_loader(
        self,
        store: RowStore,
        planner: Planner,
        *,
        micro_batch_size: int,
        num_workers: int,
    ) -> None:
        """``_stream`` + ``_dataloader`` over ``planner``, every epoch a whole
        number of optimizer updates."""
        rows_per_update = self.grad_acc_steps * micro_batch_size
        self._stream = RowStream(
            store,
            planner,
            self.rank,
            self.world_size,
            multiple_of=rows_per_update,
        )
        self._dataloader = build_loader(
            self._stream, batch_size=micro_batch_size, num_workers=num_workers
        )
        if self.steps_per_epoch == 0:
            raise ValueError(
                f"{len(store)} rows over {self.world_size} ranks give fewer rows per "
                f"rank than one optimizer update needs ({rows_per_update}); use a "
                "larger dataset or lower train_batch_size."
            )

    def epoch_microbatches(self) -> Iterator[tuple[int, list[Row], bool]]:
        """``(epoch, items, is_sync_step)`` until ``_current_step`` reaches
        ``train_steps``; the caller runs the optimizer update on every sync step
        and increments ``_current_step`` there.

        The epoch is re-derived from the step count on every pass: an iterator
        restored from a checkpoint taken on an epoch's last update yields nothing
        and the next pass starts the following epoch's plan; one restored
        mid-epoch continues on an update boundary, so counting microbatches from
        0 keeps the sync steps aligned.
        """
        while self._current_step < self.train_steps:
            epoch = self.current_epoch
            # Rows are pickled into the workers when the iterator starts.
            self._stream.set_epoch(epoch)
            for i, items in enumerate(self._dataloader):
                is_sync_step = (i + 1) % self.grad_acc_steps == 0
                yield epoch, items, is_sync_step
                if is_sync_step and self._current_step >= self.train_steps:
                    return
