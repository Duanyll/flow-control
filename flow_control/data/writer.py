"""Random cache writer (the preprocess sink) and ``finalize_cache`` (design §4).

Layout::

    <cache>/
      meta.json          # {"format": "random", "backend": ..., "version": 1, "rows": N, ...}
      index.jsonl        # merged from index-<worker>.jsonl at finalize
      rows/<hh>/<key>.pt # backend=directory, hh = sha1(key)[:2]
      data.mdb           # backend=lmdb
"""

import glob
import hashlib
import io
import os
import shutil
from typing import Any, Literal

import lmdb
import torch
from pydantic import BaseModel, ConfigDict

from flow_control.data.index import Index, IndexEntry, read_index_entries
from flow_control.data.rows import (
    COST,
    IMAGE_SIZE,
    KEY,
    KEY_PATTERN,
    Row,
    shape_signature,
)
from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import DataSink

logger = get_logger(__name__)

ROWS_DIR = "rows"
LMDB_MAP_SIZE = 1 << 40
CACHE_FORMAT_VERSION = 1

CacheBackend = Literal["directory", "lmdb"]


def prepare_output_dir(path: str, *, overwrite: bool) -> None:
    """Create ``path`` empty; refuse to clobber a non-empty one unless ``overwrite``."""
    if os.path.isdir(path) and os.listdir(path):
        if not overwrite:
            raise FileExistsError(
                f"Output {path} exists and is not empty; set overwrite=true to replace it."
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


class CacheOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    backend: CacheBackend = "directory"
    overwrite: bool = False

    def prepare(self) -> str:
        """Make ``path`` an empty directory (see ``prepare_output_dir``) and return it."""
        prepare_output_dir(self.path, overwrite=self.overwrite)
        return self.path


def directory_row_loc(key: str) -> str:
    return f"{ROWS_DIR}/{hashlib.sha1(key.encode()).hexdigest()[:2]}/{key}.pt"


class _DirectoryBackend:
    def __init__(self, path: str):
        self.path = path
        self._made: set[str] = set()

    def put(self, key: str, row: Row) -> str:
        loc = directory_row_loc(key)
        full = os.path.join(self.path, loc)
        parent = os.path.dirname(full)
        if parent not in self._made:
            os.makedirs(parent, exist_ok=True)
            self._made.add(parent)
        torch.save(row, full)
        return loc

    def close(self) -> None:
        pass


class _LmdbBackend:
    def __init__(self, path: str):
        # Default lock=True: several sink workers may write the same environment.
        self.env = lmdb.open(path, map_size=LMDB_MAP_SIZE)

    def put(self, key: str, row: Row) -> str:
        buffer = io.BytesIO()
        torch.save(row, buffer)
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), buffer.getvalue())
        return key

    def close(self) -> None:
        self.env.close()


class RandomCacheWriter(DataSink):
    """Pipeline sink: one instance per sink worker. Rows go to the backend as they
    arrive; the worker's index lines are held in memory and written to
    ``index-<worker>.jsonl`` once, at ``cleanup``."""

    def __init__(self, worker_id: int, path: str, backend: CacheBackend = "directory"):
        self.worker_id = worker_id
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._backend = (
            _DirectoryBackend(path) if backend == "directory" else _LmdbBackend(path)
        )
        self._index_lines: list[str] = []
        logger.info(f"RandomCacheWriter[{worker_id}] writing {backend} cache at {path}")

    def write(self, item: Row) -> bool:
        key = item[KEY]
        if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
            raise ValueError(
                f"Row key {key!r} is not a valid cache key ({KEY_PATTERN.pattern}); "
                "set reassign_keys=true in the preprocess config."
            )
        if COST not in item:
            raise KeyError(
                f"Row {key!r} has no {COST!r} field; RandomCacheWriter expects "
                "preprocessed rows (ProcessorStage sets it)."
            )
        image_size = item.get(IMAGE_SIZE)
        loc = self._backend.put(key, item)
        entry = IndexEntry(
            key=key,
            cost=int(item[COST]),
            sig=shape_signature(item),
            image_size=(int(image_size[0]), int(image_size[1])) if image_size else None,
            loc=loc,
        )
        self._index_lines.append(entry.to_json())
        return True

    def cleanup(self) -> None:
        self._backend.close()
        part = os.path.join(self.path, f"index-{self.worker_id}.jsonl")
        with open(part, "w", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in self._index_lines))


def finalize_cache(path: str, *, meta: dict[str, Any]) -> Index:
    """Main-process step after ``pipeline.run()``: merge ``index-*.jsonl`` (sorted
    by key), reject duplicate keys, write ``meta.json`` + ``index.jsonl``.

    ``meta`` must carry ``"backend"``; ``format`` / ``version`` / ``rows`` are set here.
    """
    if meta.get("backend") not in ("directory", "lmdb"):
        raise ValueError(
            f"finalize_cache meta['backend'] must be 'directory' or 'lmdb', got {meta.get('backend')!r}"
        )
    parts = sorted(glob.glob(os.path.join(path, "index-*.jsonl")))
    if not parts:
        raise FileNotFoundError(
            f"No index-*.jsonl under {path}: no sink worker wrote anything."
        )
    entries: list[IndexEntry] = []
    for part in parts:
        entries.extend(read_index_entries(part))
    entries.sort(key=lambda e: e.key)

    duplicates = [
        a.key for a, b in zip(entries, entries[1:], strict=False) if a.key == b.key
    ]
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} duplicate keys in {path} (e.g. {duplicates[:5]}); keys "
            "must be unique within a dataset. Set reassign_keys=true in the "
            "preprocess config, or fix the source."
        )

    index = Index(
        entries,
        {
            "format": "random",
            "version": CACHE_FORMAT_VERSION,
            **meta,
            "rows": len(entries),
        },
    )
    index.save(path)
    for part in parts:
        os.remove(part)
    logger.info(f"Finalized {meta['backend']} cache at {path}: {len(entries)} rows")
    return index
