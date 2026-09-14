"""Cross-module locks for the plan / stream layer of the data stack (design §6, §7.2,
§7.3, §9.3).

``test_grouping_plans_and_stream``: every ``groups_*`` planner yields groups of
exactly ``n`` rows, visits each row once per epoch (padding aside, and padding only
repeats rows of its own group), is deterministic per ``(seed, epoch)`` and differs
between epochs; ``rank_rows`` gives every rank the same row count with disjoint
real rows and refuses ``n % world_size != 0`` (unequal counts would desync FSDP
collectives); ``groups_shuffled`` groups are contiguous windows of a sorted
megabatch (so the in-group cost spread is the spread of ``n`` sorted neighbours);
``groups_packed`` never crosses a shard and reads each shard sequentially; and a
``RowStream`` + ``build_loader`` pair on a real ``DirectoryStore`` with
``world_size=2`` (one rank through a worker process) delivers per-batch rows from
one group, flags padding, covers the dataset, and resumes from a
``StatefulDataLoader`` state mid-epoch.

``test_row_cursor``: chunked takes never repeat within a pass, cover the dataset,
stay duplicate-free when one take straddles two passes (§15, 14.3), and resume
identically from ``state_dict``; independent takes are deterministic per
``(seed, epoch)``, repeat-free within an epoch, and rejected on a packed store;
``count > N`` raises.

``test_expand_rollouts``: the strided split gives every rank the same number of
``(prompt, k)`` pairs whose union is the full product, keeps all ``K`` rollouts of
a prompt on one rank under ``whole_prompts``, and lands the ``i``-th element of
every rank in the same group of the sequence the cursor walked.
"""

import os
import tempfile
import unittest
from collections.abc import Callable
from functools import partial

import torch

from flow_control.data import (
    COST,
    KEY,
    PADDING,
    DirectoryStore,
    Group,
    Index,
    IndexEntry,
    PackedStore,
    RandomCacheWriter,
    RowCursor,
    RowStream,
    build_loader,
    expand_rollouts,
    finalize_cache,
    groups_packed,
    groups_plain,
    groups_shuffled,
    groups_sorted,
    open_cache,
    rank_rows,
    sort_key,
)


def make_index(n_rows: int, seed: int, shard_size: int | None = None) -> Index:
    """Random small costs (many ties) and sigs; ``shard_size`` builds a packed-style
    index whose shards are ``(cost, sig)``-sorted runs, as ``pack`` writes them."""
    g = torch.Generator().manual_seed(seed)
    costs = torch.randint(1, 6, (n_rows,), generator=g).tolist()
    sigs = [f"x:{int(v)}" for v in torch.randint(0, 3, (n_rows,), generator=g)]
    keys = [(costs[i], sigs[i]) for i in range(n_rows)]
    if shard_size is None:
        order = list(range(n_rows))
    else:
        order = [
            i
            for start in range(0, n_rows, shard_size)
            for i in sorted(
                range(start, min(start + shard_size, n_rows)), key=lambda i: keys[i]
            )
        ]
    entries = [
        IndexEntry(
            f"k{i:03d}",
            costs[i],
            sigs[i],
            None,
            (pos // shard_size, 0, 0) if shard_size else f"rows/{i}.pt",
        )
        for pos, i in enumerate(order)
    ]
    return Index(entries, {})


def write_store(path: str, n_rows: int) -> DirectoryStore:
    writer = RandomCacheWriter(0, path, "directory")
    for i in range(n_rows):
        writer.write(
            {KEY: f"k{i:03d}", COST: 1 + i % 4, "v": torch.full((2,), float(i))}
        )
    writer.cleanup()
    finalize_cache(path, meta={"backend": "directory"})
    store = open_cache(path)
    assert isinstance(store, DirectoryStore)
    return store


def real_rows(groups: list[Group]) -> list[int]:
    return [row_id for g in groups for row_id, pad in g if not pad]


class DataGroupingTest(unittest.TestCase):
    def check_plan(self, groups: list[Group], n: int, n_rows: int) -> None:
        self.assertTrue(all(len(g) == n for g in groups))
        self.assertEqual(sorted(real_rows(groups)), list(range(n_rows)))
        for g in groups:
            own = {row_id for row_id, pad in g if not pad}
            self.assertTrue(all(row_id in own for row_id, pad in g if pad))
        for world_size in (2, 4):
            per_rank = [rank_rows(groups, r, world_size) for r in range(world_size)]
            self.assertEqual(
                {len(rows) for rows in per_rank}, {len(groups) * n // world_size}
            )
            seen: set[int] = set()
            for rows in per_rank:
                mine = {row_id for row_id, pad in rows if not pad}
                self.assertFalse(mine & seen)
                seen |= mine
            self.assertEqual(seen, set(range(n_rows)))

    def test_grouping_plans_and_stream(self):
        n, k, n_rows = 4, 3, 21
        index = make_index(n_rows, seed=0)
        packed = make_index(n_rows, seed=0, shard_size=6)
        planners: dict[str, Callable[[int], list[Group]]] = {
            "shuffled": partial(groups_shuffled, index, n, k, 5),
            "sorted": lambda epoch: groups_sorted(index, n),
            "packed": partial(groups_packed, packed, n, 5),
            "packed_inner": partial(
                groups_packed, packed, n, 5, shuffle_within_shard=True
            ),
            "plain": partial(groups_plain, n_rows, n, 5, shuffle=True),
            "plain_ordered": partial(groups_plain, n_rows, n, 5, shuffle=False),
        }
        for name, planner in planners.items():
            with self.subTest(planner=name):
                groups = planner(0)
                self.check_plan(groups, n, n_rows)
                self.assertEqual(groups, planner(0))
                if name in ("sorted", "plain_ordered"):
                    self.assertEqual(groups, planner(1))
                else:
                    self.assertNotEqual(groups, planner(1))

        with self.assertRaisesRegex(ValueError, "multiple of world_size"):
            rank_rows(planners["shuffled"](0), 0, 3)

        # groups_shuffled: each group is a contiguous window of one sorted
        # megabatch (§6.2 formula), so its cost spread is that of n sorted neighbours.
        g = torch.Generator().manual_seed(5 * 1_000_003 + 0)
        perm = torch.randperm(n_rows, generator=g).tolist()
        megabatches = [
            sorted(perm[s : s + n * k], key=lambda i: sort_key(index, i))
            for s in range(0, n_rows, n * k)
        ]
        for group in planners["shuffled"](0):
            rows = [row_id for row_id, pad in group if not pad]
            mega = next(m for m in megabatches if rows[0] in m)
            pos = mega.index(rows[0])
            self.assertEqual(mega[pos : pos + len(rows)], rows)

        ordered = real_rows(planners["sorted"](0))
        self.assertEqual(ordered, sorted(ordered, key=lambda i: sort_key(index, i)))
        self.assertEqual(real_rows(planners["plain_ordered"](0)), list(range(n_rows)))

        # groups_packed: one shard per group, read as a contiguous run of row ids.
        shard_of = [e.loc[0] for e in packed.entries]
        for name in ("packed", "packed_inner"):
            for group in planners[name](0):
                rows = [row_id for row_id, pad in group if not pad]
                self.assertEqual(len({shard_of[r] for r in rows}), 1)
                self.assertEqual(rows, list(range(rows[0], rows[0] + len(rows))))
        with self.assertRaisesRegex(TypeError, "packed cache index"):
            groups_packed(index, n, 0, 0)

        # End to end: RowStream + build_loader on a DirectoryStore, world_size=2,
        # rank 0 through a worker process, batch = n / world_size.
        n, k, n_rows, world_size = 4, 2, 10, 2
        with tempfile.TemporaryDirectory() as tmp:
            store = write_store(os.path.join(tmp, "cache"), n_rows)
            planner = partial(groups_shuffled, store.index, n, k, 3)
            streams = [
                RowStream(store, planner, r, world_size) for r in range(world_size)
            ]
            loaders = [
                build_loader(streams[r], batch_size=n // world_size, num_workers=1 - r)
                for r in range(world_size)
            ]
            row_id_of = {e.key: i for i, e in enumerate(store.index.entries)}
            for epoch in range(2):
                groups = planner(epoch)
                group_of = {
                    row_id: gi for gi, g in enumerate(groups) for row_id, _ in g
                }
                for stream in streams:
                    stream.set_epoch(epoch)
                self.assertEqual({len(loader) for loader in loaders}, {len(groups)})
                covered: set[str] = set()
                for rank, loader in enumerate(loaders):
                    planned = rank_rows(groups, rank, world_size)
                    got: list[tuple[int, bool]] = []
                    for gi, batch in enumerate(loader):
                        self.assertEqual(
                            {group_of[row_id_of[r[KEY]]] for r in batch}, {gi}
                        )
                        got.extend(
                            (row_id_of[r[KEY]], r.get(PADDING, False)) for r in batch
                        )
                        covered |= {r[KEY] for r in batch if not r.get(PADDING)}
                    self.assertEqual(got, planned)
                self.assertEqual(covered, {e.key for e in store.index.entries})

            # Resume mid-epoch from the dataloader state (the only cursor there is).
            loader = loaders[0]
            it = iter(loader)
            next(it)
            state = loader.state_dict()
            rest = [[r[KEY] for r in batch] for batch in it]
            resumed = build_loader(
                streams[0], batch_size=n // world_size, num_workers=1
            )
            resumed.load_state_dict(state)
            self.assertEqual([[r[KEY] for r in batch] for batch in resumed], rest)
            self.assertEqual(len(rest), len(loader) - 1)

    def test_row_cursor(self):
        n_rows = 10
        with tempfile.TemporaryDirectory() as tmp:
            store = write_store(os.path.join(tmp, "cache"), n_rows)
            planner = partial(
                groups_shuffled, store.index, 4, 2, 11
            )  # 3 groups, 2 padding

            cursor = RowCursor(store, planner, seed=0)
            self.assertEqual(sorted(cursor.take(n_rows, epoch=0)), list(range(n_rows)))

            cursor = RowCursor(store, planner, seed=0)
            first = cursor.take(6, epoch=0)
            second = cursor.take(6, epoch=1)  # 4 left in pass 0 + 2 from pass 1
            self.assertEqual(len(set(first)), 6)
            self.assertEqual(len(set(second)), 6)
            self.assertEqual(set(first) | set(second[:4]), set(range(n_rows)))
            self.assertEqual(cursor.state_dict()["pass"], 1)

            state = cursor.state_dict()
            expected = [cursor.take(5, epoch=e) for e in (2, 3)]
            resumed = RowCursor(store, planner, seed=0)
            resumed.load_state_dict(state)
            self.assertEqual([resumed.take(5, epoch=e) for e in (2, 3)], expected)
            with self.assertRaisesRegex(ValueError, "distinct prompts"):
                cursor.take(n_rows + 1, epoch=4)

            independent = RowCursor(store, planner, seed=0, sampling="independent")
            first = independent.take(n_rows, epoch=0)
            self.assertEqual(sorted(first), list(range(n_rows)))
            self.assertEqual(
                RowCursor(store, planner, seed=0, sampling="independent").take(
                    n_rows, epoch=0
                ),
                first,
            )
            self.assertNotEqual(independent.take(n_rows, epoch=1), first)
            self.assertNotEqual(
                RowCursor(store, planner, seed=1, sampling="independent").take(
                    n_rows, epoch=0
                ),
                first,
            )
            with self.assertRaisesRegex(ValueError, "packed cache"):
                RowCursor(
                    PackedStore(tmp, store.index),
                    planner,
                    seed=0,
                    sampling="independent",
                )

    def test_expand_rollouts(self):
        n, k, n_rows, M, K = 4, 2, 12, 8, 3
        with tempfile.TemporaryDirectory() as tmp:
            store = write_store(os.path.join(tmp, "cache"), n_rows)
            planner = partial(groups_shuffled, store.index, n, k, 2)
            groups = planner(0)
            group_of = {row_id: gi for gi, g in enumerate(groups) for row_id, _ in g}
            prompt_ids = RowCursor(store, planner, seed=0).take(M, epoch=0)
        all_pairs = {(p, kk) for p in prompt_ids for kk in range(K)}
        for world_size in (2, 4):
            for whole in (False, True):
                with self.subTest(world_size=world_size, whole_prompts=whole):
                    shares = [
                        expand_rollouts(
                            prompt_ids, K, r, world_size, whole_prompts=whole
                        )
                        for r in range(world_size)
                    ]
                    self.assertEqual({len(s) for s in shares}, {M * K // world_size})
                    self.assertEqual(
                        sorted(pair for s in shares for pair in s), sorted(all_pairs)
                    )
                    for i in range(M * K // world_size):
                        self.assertEqual(len({group_of[s[i][0]] for s in shares}), 1)
                    if whole:
                        for share in shares:
                            for p in {p for p, _ in share}:
                                self.assertEqual(
                                    [kk for q, kk in share if q == p], list(range(K))
                                )


if __name__ == "__main__":
    unittest.main()
