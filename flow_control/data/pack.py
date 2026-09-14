"""``pack``: random cache -> packed cache of tar shards (design §5).

Layout::

    <packed>/
      meta.json          # {"format": "packed", "version": 1, "shard_size": S, "group_size": n,
                         #  "seed": s, "source": ..., "shards": K, "rows": N, ...}
      index.jsonl        # shard order; loc = (shard, offset, size), offset -> member data
      shards/000000.tar  # members <key>.pt in (cost, sig) order within the shard

Each shard is a megabatch: a random slice of the source, sorted by ``(cost, sig)``.
The last shard may be short; padding is a read-time plan concept, never bytes.
"""

import io
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch
from pydantic import BaseModel, ConfigDict
from rich.progress import Progress

from flow_control.data.index import Index, IndexEntry
from flow_control.data.store import SHARDS_DIR, RowStore, open_cache, shard_path
from flow_control.data.writer import CACHE_FORMAT_VERSION, prepare_output_dir
from flow_control.utils.logging import console, get_logger

logger = get_logger(__name__)

ShardMember = tuple[int, int, int]
"""``(row_id, offset, size)`` of one tar member; offset points at the member data."""


class PackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str
    """Random cache directory written by `flow-control preprocess`."""
    output: str
    shard_size: int = 480
    """Rows per shard; should be a multiple of group_size (480 = lcm(16, 24) * 10)."""
    group_size: int = 16
    """Only checked against shard_size and recorded in meta; grouping happens at read time."""
    seed: int = 0
    num_workers: int = 8
    overwrite: bool = False


_store: RowStore | None = None


def _init_worker(store: RowStore) -> None:
    global _store
    _store = store


def _write_shard(shard: int, row_ids: list[int], tar_path: str) -> list[ShardMember]:
    assert _store is not None, "_init_worker was not called"
    members: list[ShardMember] = []
    with tarfile.open(tar_path, "w") as tf:
        for row_id in row_ids:
            entry = _store.index.entries[row_id]
            buffer = io.BytesIO()
            torch.save(_store.get(row_id), buffer)
            data = buffer.getvalue()
            info = tarfile.TarInfo(f"{entry.key}.pt")
            info.size = len(data)
            # tf.offset is the header position; the data starts right after the
            # header block(s) addfile will emit (pax headers included).
            offset = tf.offset + len(info.tobuf(tf.format, tf.encoding, tf.errors))
            tf.addfile(info, io.BytesIO(data))
            members.append((row_id, offset, len(data)))
    return members


def plan_shards(index: Index, shard_size: int, seed: int) -> list[list[int]]:
    """Seeded permutation cut into shards, each sorted by ``(cost, sig)``."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(index), generator=generator).tolist()
    return [
        sorted(
            perm[start : start + shard_size],
            key=lambda i: (index.entries[i].cost, index.entries[i].sig),
        )
        for start in range(0, len(perm), shard_size)
    ]


def _run_shards(
    store: RowStore, jobs: list[tuple[int, list[int], str]], num_workers: int
) -> list[list[ShardMember]]:
    results: list[list[ShardMember]] = [[] for _ in jobs]
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Packing shards", total=len(jobs))
        if num_workers <= 1:
            _init_worker(store)
            for shard, row_ids, path in jobs:
                results[shard] = _write_shard(shard, row_ids, path)
                progress.advance(task)
            return results
        with ProcessPoolExecutor(
            max_workers=min(num_workers, len(jobs)),
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(store,),
        ) as pool:
            futures = {
                pool.submit(_write_shard, shard, row_ids, path): shard
                for shard, row_ids, path in jobs
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                progress.advance(task)
    return results


def pack(config: PackConfig) -> Index:
    store = open_cache(config.input)
    index = store.index
    if index.meta.get("format") != "random":
        raise ValueError(
            f"{config.input} is a {index.meta.get('format')!r} cache; pack expects a "
            "random cache written by `flow-control preprocess`."
        )
    if len(index) == 0:
        raise ValueError(f"{config.input} is empty; nothing to pack.")
    if config.shard_size % config.group_size != 0:
        logger.warning(
            f"shard_size={config.shard_size} is not a multiple of group_size="
            f"{config.group_size}; every shard but the last will end with a partial "
            "group that the reader pads."
        )

    prepare_output_dir(config.output, overwrite=config.overwrite)
    os.makedirs(os.path.join(config.output, SHARDS_DIR))
    shards = plan_shards(index, config.shard_size, config.seed)
    jobs = [
        (shard, row_ids, shard_path(config.output, shard))
        for shard, row_ids in enumerate(shards)
    ]
    results = _run_shards(store, jobs, config.num_workers)

    entries = [
        IndexEntry(
            index.entries[row_id].key,
            index.entries[row_id].cost,
            index.entries[row_id].sig,
            index.entries[row_id].image_size,
            (shard, offset, size),
        )
        for shard, members in enumerate(results)
        for row_id, offset, size in members
    ]
    meta = {
        k: v
        for k, v in index.meta.items()
        if k not in ("format", "backend", "version", "rows")
    }
    meta.update(
        format="packed",
        version=CACHE_FORMAT_VERSION,
        shard_size=config.shard_size,
        group_size=config.group_size,
        seed=config.seed,
        source=os.path.abspath(config.input),
        shards=len(shards),
        rows=len(entries),
    )
    packed = Index(entries, meta)
    packed.save(config.output)
    logger.info(
        f"Packed {len(entries)} rows from {config.input} into {len(shards)} shards at {config.output}"
    )
    return packed
