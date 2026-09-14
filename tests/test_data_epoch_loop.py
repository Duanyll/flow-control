"""Cross-module lock for the step-counted epoch loop: ``EpochLoopMixin`` over
``RowStream`` (``multiple_of``) + ``StatefulDataLoader`` resume.

Every optimizer update is exactly ``grad_acc_steps`` microbatches of one epoch's
plan, in a fresh run and in a run resumed from a checkpoint taken after any
update, and the resumed run reaches ``train_steps`` with the same microbatch
sequence as the uninterrupted one. Before this lock the epoch's tail (rows short
of one update) was skipped by the ``enumerate`` index, which restarts at 0 on a
resumed iterator: a run resumed mid-epoch trained the tail without a sync step
and leaked its gradients into the next epoch's first update, and a checkpoint
taken on an epoch's last update resumed into an exhausted iterator that burnt one
iteration of the ``range(starting_epoch, total_epochs)`` loop, ending the run
short of ``train_steps``.
"""

import unittest
from functools import partial
from typing import Any

import torch

from flow_control.data import KEY, Index, IndexEntry, Row, groups_plain, rank_rows
from flow_control.training.mixins import EpochLoopMixin

N_ROWS = 14
MICRO, GRAD_ACC, TRAIN_STEPS = 2, 2, 7
# 14 rows -> 7 groups of 2 (world_size 1) -> 14 rows, truncated to 12 (one
# update = 4 rows) -> 6 microbatches = 3 updates per epoch; 7 steps span 3 epochs.


class _ListStore:
    """Picklable stand-in for a RowStore (workers fork it)."""

    index = Index([IndexEntry(str(i), 0, "", None, str(i)) for i in range(N_ROWS)], {})

    def __len__(self) -> int:
        return N_ROWS

    def get(self, row_id: int) -> Row:
        return {KEY: str(row_id), "v": torch.tensor(float(row_id))}


class _Probe(EpochLoopMixin):
    launch: Any = None
    train_steps: int = TRAIN_STEPS

    @property
    def grad_acc_steps(self) -> int:
        return GRAD_ACC


Record = tuple[int, list[str], bool]
"""``(epoch, keys, is_sync_step)`` of one microbatch."""


def _run(
    probe: _Probe, *, stop_after: int | None = None
) -> tuple[list[Record], dict[str, Any] | None]:
    """Drive the loop like a trainer: bump the step on sync steps; when
    ``stop_after`` steps are done, snapshot the loader (as ``save_maybe`` would,
    before the iterator advances) and stop."""
    records: list[Record] = []
    for epoch, items, is_sync_step in probe.epoch_microbatches():
        records.append((epoch, [row[KEY] for row in items], is_sync_step))
        if not is_sync_step:
            continue
        probe._current_step += 1
        if probe._current_step == stop_after:
            return records, probe._dataloader.state_dict()
    return records, None


class EpochLoopResumeTest(unittest.TestCase):
    def test_updates_are_whole_and_resume_on_update_boundaries(self) -> None:
        store: Any = _ListStore()
        planner = partial(groups_plain, N_ROWS, 2, 0, shuffle=True)

        def make(num_workers: int) -> _Probe:
            probe = _Probe()
            probe.make_train_loader(
                store, planner, micro_batch_size=MICRO, num_workers=num_workers
            )
            return probe

        fresh = make(0)
        self.assertEqual((len(fresh._stream), fresh.steps_per_epoch), (12, 3))
        full, _ = _run(fresh)
        self.assertEqual(fresh._current_step, TRAIN_STEPS)
        self.assertEqual(len(full), TRAIN_STEPS * GRAD_ACC)
        self.assertEqual([e for e, _, _ in full], [0] * 6 + [1] * 6 + [2] * 2)
        self.assertEqual([s for _, _, s in full], [False, True] * TRAIN_STEPS)
        # Each epoch's microbatches are the first 12 rows of its plan, in order;
        # the two rows the plan leaves over differ between epochs.
        for epoch in range(3):
            planned = [str(r) for r, _ in rank_rows(planner(epoch), 0, 1)]
            fed = [k for e, keys, _ in full if e == epoch for k in keys]
            self.assertEqual(fed, planned[: len(fed)])
        self.assertNotEqual(
            rank_rows(planner(0), 0, 1)[12:], rank_rows(planner(1), 0, 1)[12:]
        )

        # Resume after step 1 (mid-epoch), 3 (last update of epoch 0: the
        # restored iterator is exhausted), 4 (first update of epoch 1); the
        # epoch-end case also through a worker process.
        for stop_after, num_workers in ((1, 0), (3, 0), (3, 1), (4, 0)):
            with self.subTest(stop_after=stop_after, num_workers=num_workers):
                head, snapshot = _run(make(num_workers), stop_after=stop_after)
                assert snapshot is not None
                resumed = make(num_workers)
                resumed._dataloader.load_state_dict(snapshot)
                resumed._current_step = stop_after
                tail, _ = _run(resumed)
                self.assertEqual(resumed._current_step, TRAIN_STEPS)
                self.assertEqual(head + tail, full)

        # Fewer rows per rank than one update: refused at loader creation.
        with self.assertRaisesRegex(ValueError, "fewer rows per rank"):
            _Probe().make_train_loader(
                store,
                partial(groups_plain, N_ROWS, 2, 0, shuffle=True),
                micro_batch_size=N_ROWS,
                num_workers=0,
            )


if __name__ == "__main__":
    unittest.main()
