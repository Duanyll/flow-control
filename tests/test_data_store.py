"""Cross-module locks for the cache storage layer (design §4-§5, §7.1).

``test_random_and_packed_stores_agree``: the same synthetic rows written through
``RandomCacheWriter`` (directory and lmdb, split across two sink workers),
finalized, packed, and reopened via ``open_store`` come back field-for-field
identical from all three stores; ``PackedStore``'s seek reads match a plain
sequential ``tarfile`` walk (so ``loc`` offsets really point at member data, not
headers); stores survive pickling (DataLoader workers); ``limit`` truncates;
``finalize_cache`` rejects duplicate keys with the ``reassign_keys`` hint; and the
online path coerces a raw source through the processor's input TypedDict.

``test_pack_shard_layout``: every shard is ``(cost, sig)``-monotone, ``index.jsonl``
order equals tar member order, the tail shard is short and unpadded, and a
``shard_size`` that is not a multiple of ``group_size`` only warns (§15, 14.5).
"""

import os
import pickle
import tarfile
import tempfile
import unittest
from typing import Any

import torch

from flow_control.data import (
    COST,
    IMAGE_SIZE,
    KEY,
    DirectoryStore,
    Index,
    IndexEntry,
    LmdbStore,
    OnlineStore,
    PackConfig,
    PackedStore,
    RandomCacheWriter,
    RowStore,
    finalize_cache,
    open_store,
    pack,
)
from flow_control.data.store import shard_path
from flow_control.processors.tasks.t2i import T2IProcessor


def make_rows(n: int, seed: int) -> list[dict[str, Any]]:
    """Small varied rows: cost ties across different shapes exercise the sig tie-break."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        h, w = (int(v) for v in torch.randint(1, 4, (2,), generator=g))
        text = int(torch.randint(2, 5, (1,), generator=g))
        rows.append(
            {
                KEY: f"row-{i:03d}",
                COST: h * w + text,
                IMAGE_SIZE: (h * 8, w * 8),
                "clean_latents": torch.randn(2, 4, h, w, generator=g),
                "prompt_embeds": torch.randn(1, text, 8, generator=g).to(
                    torch.bfloat16
                ),
                "reference_latents": [
                    torch.randn(1, 4, 2, 2, generator=g) for _ in range(i % 3)
                ],
                "prompt": f"prompt {i}",
                "extra": {"nested": i, "tags": ["a", "b"]},
            }
        )
    return rows


def write_cache(rows, path: str, backend, workers: int = 2) -> Index:
    writers = [RandomCacheWriter(w, path, backend) for w in range(workers)]
    for i, row in enumerate(rows):
        writers[i % workers].write(row)
    for writer in writers:
        writer.cleanup()
    return finalize_cache(path, meta={"backend": backend, "mode": "training"})


def rows_by_key(store: RowStore) -> dict[str, dict[str, Any]]:
    return {store.index.entries[i].key: store.get(i) for i in range(len(store))}


def packed_loc(entry: IndexEntry) -> tuple[int, int, int]:
    assert isinstance(entry.loc, tuple)
    return entry.loc


class DataStoreTest(unittest.TestCase):
    def assert_row_equal(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        self.assertEqual(set(a), set(b))
        for name, va in a.items():
            vb = b[name]
            if isinstance(va, torch.Tensor):
                self.assertEqual(va.dtype, vb.dtype, name)
                self.assertTrue(torch.equal(va, vb), name)
            elif isinstance(va, list) and va and isinstance(va[0], torch.Tensor):
                self.assertEqual(len(va), len(vb), name)
                for x, y in zip(va, vb, strict=True):
                    self.assertTrue(torch.equal(x, y), name)
            else:
                self.assertEqual(va, vb, name)

    def open_cache(self, path: str, **extra: Any) -> RowStore:
        return open_store(
            {"type": "cache", "path": path, **extra},
            processor_class=T2IProcessor,
            mode="training",
        )

    def test_random_and_packed_stores_agree(self):
        rows = make_rows(30, seed=0)
        expected = {row[KEY]: row for row in rows}
        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                name: os.path.join(tmp, name)
                for name in ("directory", "lmdb", "packed")
            }
            write_cache(rows, paths["directory"], "directory")
            # py-lmdb refuses to open one environment twice in a process; real sink
            # workers are separate processes, so lmdb gets a single writer here.
            write_cache(rows, paths["lmdb"], "lmdb", workers=1)
            pack(
                PackConfig(
                    input=paths["directory"],
                    output=paths["packed"],
                    shard_size=8,
                    num_workers=2,
                )
            )

            stores = {name: self.open_cache(path) for name, path in paths.items()}
            self.assertIsInstance(stores["directory"], DirectoryStore)
            self.assertIsInstance(stores["lmdb"], LmdbStore)
            self.assertIsInstance(stores["packed"], PackedStore)
            for name, store in stores.items():
                with self.subTest(store=name):
                    self.assertEqual(len(store), len(rows))
                    got = rows_by_key(store)
                    self.assertEqual(set(got), set(expected))
                    for key, row in got.items():
                        self.assert_row_equal(row, expected[key])
                    # get() hands out independent dicts
                    first = store.get(0)
                    first["mutated"] = True
                    self.assertNotIn("mutated", store.get(0))
                    # Pickled into a DataLoader worker: handles reopen lazily. The
                    # original is closed first because LMDB allows one open
                    # environment per path per process (workers are processes).
                    blob = pickle.dumps(store)
                    if isinstance(store, LmdbStore | PackedStore):
                        store.close()
                    clone = pickle.loads(blob)
                    last = len(store) - 1
                    self.assert_row_equal(
                        clone.get(last), expected[clone.index.entries[last].key]
                    )

            # Packed seek reads == sequential tarfile walk; loc offsets hit member data.
            packed = stores["packed"]
            row_ids = {e.key: i for i, e in enumerate(packed.index.entries)}
            for shard in range(packed.index.meta["shards"]):
                with tarfile.open(shard_path(paths["packed"], shard)) as tf:
                    for member in tf.getmembers():
                        row_id = row_ids[member.name.removesuffix(".pt")]
                        extracted = tf.extractfile(member)
                        assert extracted is not None
                        sequential = torch.load(extracted, weights_only=True)
                        self.assert_row_equal(packed.get(row_id), sequential)
                        _, offset, size = packed_loc(packed.index.entries[row_id])
                        self.assertEqual(
                            (offset, size), (member.offset_data, member.size)
                        )

            self.assertEqual(len(self.open_cache(paths["packed"], limit=5)), 5)

            with self.assertRaisesRegex(ValueError, "reassign_keys"):
                write_cache(rows[:3] + rows[:1], os.path.join(tmp, "dup"), "directory")

        online = open_store(
            {"type": "inline", "data": [{"prompt": "p", "image_size": "[64, 64]"}]},
            processor_class=T2IProcessor,
            mode="inference",
        )
        self.assertIsInstance(online, OnlineStore)
        row = online.get(0)
        self.assertEqual(
            (row[KEY], row["prompt"], tuple(row["image_size"])), ("0", "p", (64, 64))
        )

    def test_pack_shard_layout(self):
        rows = make_rows(21, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            random_path = os.path.join(tmp, "random")
            packed_path = os.path.join(tmp, "packed")
            write_cache(rows, random_path, "directory", workers=1)

            with self.assertLogs("flow_control.data.pack", level="WARNING") as logs:
                pack(
                    PackConfig(
                        input=random_path,
                        output=packed_path,
                        shard_size=6,
                        group_size=4,
                        num_workers=1,
                    )
                )
            self.assertTrue(any("multiple of group_size" in m for m in logs.output))

            index = Index.load(packed_path)
            self.assertEqual(
                (index.meta["format"], index.meta["shards"], len(index)),
                ("packed", 4, 21),
            )
            self.assertEqual(
                sorted(e.key for e in index.entries), sorted(r[KEY] for r in rows)
            )
            shard_of = [packed_loc(e)[0] for e in index.entries]
            self.assertEqual(shard_of, sorted(shard_of))  # index.jsonl in shard order
            for shard in range(4):
                entries = [e for e in index.entries if packed_loc(e)[0] == shard]
                # Tail shard stays short: no padding bytes at pack time.
                self.assertEqual(len(entries), 6 if shard < 3 else 3)
                sort_keys = [(e.cost, e.sig) for e in entries]
                self.assertEqual(sort_keys, sorted(sort_keys))
                with tarfile.open(shard_path(packed_path, shard)) as tf:
                    self.assertEqual(tf.getnames(), [f"{e.key}.pt" for e in entries])

            with self.assertNoLogs("flow_control.data.pack", level="WARNING"):
                pack(
                    PackConfig(
                        input=random_path,
                        output=packed_path,
                        shard_size=8,
                        group_size=4,
                        num_workers=1,
                        overwrite=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
