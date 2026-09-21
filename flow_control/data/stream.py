"""Consumer-side row access (design §7.2, §7.3).

``RowStream`` is the map-style dataset behind SFT / inference / validation: the
rows of one rank for one epoch, resolved through a planner. ``RowCursor`` is the
RL prompt cursor: ``take`` hands every rank the same ``count`` distinct row ids.
Both leave randomness to the plan; the stream itself is stateless so the only
cursor is the ``StatefulDataLoader``'s yielded count.
"""

import random
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.data.grouping import Group, PlannedRow, rank_rows
from flow_control.data.rows import PADDING, Row
from flow_control.data.store import PackedStore, RowStore
from flow_control.samplers.sampler import derive_seed

Planner = Callable[[int], list[Group]]
"""``epoch -> groups``; see ``groups_*`` in ``grouping.py``."""


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducible randomness across workers."""
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is not None
    seed = worker_info.seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def identity_collate(rows: list[Row]) -> list[Row]:
    """No stacking (design D8): every row stays an independent dict."""
    return rows


class RowStream(Dataset):
    """This rank's rows for the current epoch (map-style).

    Call ``set_epoch`` before creating the epoch's iterator: the rows are pickled
    into the DataLoader workers at that point, so it must run in the main process
    first. Padding rows come back with ``padding = True``.

    ``multiple_of``: a trainer passes its rows per optimizer update so every
    epoch is a whole number of updates; the tail short of that is dropped from
    the plan (the same count on every rank, so collectives stay balanced) and,
    since the plan is reshuffled, holds different rows every epoch.
    """

    def __init__(
        self,
        store: RowStore,
        planner: Planner,
        rank: int,
        world_size: int,
        *,
        multiple_of: int = 1,
    ):
        if multiple_of < 1:
            raise ValueError(f"multiple_of must be >= 1, got {multiple_of}")
        self.store = store
        self.planner = planner
        self.rank = rank
        self.world_size = world_size
        self.multiple_of = multiple_of
        self._rows: list[PlannedRow] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rows = rank_rows(self.planner(epoch), self.rank, self.world_size)
        self._rows = rows[: len(rows) - len(rows) % self.multiple_of]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Row:
        row_id, padding = self._rows[index]
        row = self.store.get(row_id)
        if padding:
            row[PADDING] = True
        return row


def build_loader(
    stream: RowStream, *, batch_size: int, num_workers: int
) -> StatefulDataLoader:
    return StatefulDataLoader(
        stream,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=identity_collate,
        worker_init_fn=seed_worker,
    )


PromptSampling = Literal["chunked", "independent"]


class RowCursor:
    """RL prompt cursor: one ``take`` per RL epoch, identical on every rank.

    ``chunked``: walk the group sequence of the current pass in order; a new pass
    (new plan) starts when one is exhausted, and a ``take`` that straddles two
    passes concatenates them, de-duplicating by ``row_id`` within the take.
    Prompts taken together are adjacent in the group sequence, hence similar in
    cost; ``expand_rollouts`` keeps that adjacency across ranks.

    ``independent``: every ``take`` is ``randperm(N)[:count]`` seeded by
    ``derive_seed(seed, f"prompts:{epoch}")`` -- no repeats inside an epoch,
    independent between epochs, cost-blind (baseline reproduction only). Not
    allowed on a ``PackedStore``: it would seek at random across shards.

    State is ``(pass, position)``; it goes into the trainer checkpoint.

    Alignment caveat (chunked): ``expand_rollouts`` cuts the taken prompts into
    blocks of ``world_size`` (per prompt under ``whole_prompts``, per rollout
    otherwise), and a block lies inside one plan group only while the walk has
    consumed whole groups. Skipping a padded group's padding (``N mod
    group_size`` real rows) or a duplicate at a pass boundary shifts every later
    block by that many rows, so from then on one block per group straddles two
    groups of possibly different cost. This costs throughput (a rank waits on
    the collective), never correctness; it is exact when ``(N mod group_size) %
    world_size == 0`` and irrelevant when ``K % world_size == 0`` without
    pairwise rewards (each block is then inside one prompt).
    """

    def __init__(
        self,
        store: RowStore,
        planner: Planner,
        seed: int,
        sampling: PromptSampling = "chunked",
    ):
        if sampling == "independent" and isinstance(store, PackedStore):
            raise ValueError(
                "prompt_sampling='independent' is not supported on a packed cache "
                "(random seeks across shards); use 'chunked' or a random cache."
            )
        self.store = store
        self.planner = planner
        self.seed = seed
        self.sampling: PromptSampling = sampling
        self._pass = 0
        self._position = 0
        self._pass_rows: list[PlannedRow] | None = None

    def _rows(self) -> list[PlannedRow]:
        if self._pass_rows is None:
            self._pass_rows = [row for g in self.planner(self._pass) for row in g]
        return self._pass_rows

    def take(self, count: int, *, epoch: int) -> list[int]:
        """``count`` distinct global row ids, padding skipped."""
        total = len(self.store)
        if count > total:
            raise ValueError(
                f"Cannot take {count} distinct prompts from a store of {total} rows; "
                "lower num_prompts_per_epoch or use a larger dataset."
            )
        if self.sampling == "independent":
            g = torch.Generator().manual_seed(
                derive_seed(self.seed, f"prompts:{epoch}")
            )
            return torch.randperm(total, generator=g)[:count].tolist()

        taken: list[int] = []
        seen: set[int] = set()
        while len(taken) < count:
            rows = self._rows()
            while self._position < len(rows) and len(taken) < count:
                row_id, padding = rows[self._position]
                self._position += 1
                if padding or row_id in seen:
                    continue
                seen.add(row_id)
                taken.append(row_id)
            if self._position >= len(rows):
                self._pass += 1
                self._position = 0
                self._pass_rows = None
        return taken

    def state_dict(self) -> dict[str, int]:
        return {"pass": self._pass, "position": self._position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._pass = int(state["pass"])
        self._position = int(state["position"])
        self._pass_rows = None


if __name__ == "__main__":
    from functools import partial

    from rich import print

    from flow_control.data.grouping import groups_plain
    from flow_control.data.index import Index, IndexEntry

    class ListStore:
        def __init__(self, n: int):
            self.index = Index(
                [IndexEntry(str(i), 0, "", None, str(i)) for i in range(n)], {}
            )

        def __len__(self) -> int:
            return len(self.index)

        def get(self, row_id: int) -> Row:
            return {"key": str(row_id), "v": torch.full((2,), float(row_id))}

    store: Any = ListStore(10)
    planner = partial(groups_plain, 10, 4, 7, shuffle=True)
    streams = [RowStream(store, planner, rank, 2) for rank in range(2)]
    for epoch in range(2):
        for s in streams:
            s.set_epoch(epoch)
        loader = build_loader(streams[0], batch_size=2, num_workers=0)
        batches = list(loader)
        print(epoch, [[r["key"] for r in b] for b in batches])
        assert len(batches) == len(streams[0]) // 2 == 3
        keys = {
            r["key"]
            for s in streams
            for i in range(len(s))
            for r in [s[i]]
            if not r.get(PADDING)
        }
        assert keys == {str(i) for i in range(10)}

    # multiple_of drops the tail short of one update: 6 rows per rank -> 4.
    truncated = RowStream(store, planner, 0, 2, multiple_of=4)
    assert len(truncated) == 4 and len(streams[0]) == 6

    cursor = RowCursor(store, planner, seed=0)
    first = cursor.take(6, epoch=0)
    second = cursor.take(6, epoch=1)
    print(first, second)
    assert len(set(first)) == 6 and len(set(second)) == 6
    assert set(first) | set(second[:4]) == set(range(10))
    state = cursor.state_dict()
    third = cursor.take(3, epoch=2)
    cursor.load_state_dict(state)
    assert cursor.take(3, epoch=2) == third

    independent = RowCursor(store, planner, seed=0, sampling="independent")
    a = independent.take(10, epoch=0)
    assert sorted(a) == list(range(10))
    assert a == RowCursor(store, planner, seed=0, sampling="independent").take(
        10, epoch=0
    )
    assert a != independent.take(10, epoch=1)
