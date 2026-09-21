"""``RowStore``: random access to rows by ``row_id`` (design §7.1).

Every store is picklable into DataLoader workers: file / LMDB handles are dropped
on pickle and reopened lazily in the receiving process. ``get`` always returns a
fresh ``dict`` the consumer may modify in place.
"""

import io
import os
from collections import OrderedDict
from typing import Any, Protocol, cast

import lmdb
import torch

from flow_control.data.index import Index, IndexEntry
from flow_control.data.rows import KEY, Row
from flow_control.data.sources.base import RawSource

SHARDS_DIR = "shards"
MAX_OPEN_SHARDS = 8
"""Per-process cap on simultaneously open tar handles in ``PackedStore``."""


class RowStore(Protocol):
    index: Index

    def __len__(self) -> int: ...
    def get(self, row_id: int) -> Row: ...


def _load_row(data: str | bytes | memoryview) -> Row:
    row = torch.load(
        data if isinstance(data, str) else io.BytesIO(data), weights_only=True
    )
    # Existing preprocessed caches keep their on-disk format; normalize on read.
    if "__key__" in row:
        row.setdefault(KEY, row.pop("__key__"))
    return row


class DirectoryStore:
    """backend=directory: ``torch.load(<path>/<loc>)``."""

    def __init__(self, path: str, index: Index):
        self.path = path
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def get(self, row_id: int) -> Row:
        loc = cast(str, self.index.entries[row_id].loc)
        return _load_row(os.path.join(self.path, loc))


class LmdbStore:
    """backend=lmdb: read-only, ``lock=False`` environment opened lazily."""

    def __init__(self, path: str, index: Index):
        self.path = path
        self.index = index
        self._env: lmdb.Environment | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _environment(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._env

    def get(self, row_id: int) -> Row:
        key = cast(str, self.index.entries[row_id].loc)
        with self._environment().begin(write=False) as txn:
            value = txn.get(key.encode())
        if value is None:
            raise KeyError(
                f"Key {key!r} listed in the index is missing from {self.path}"
            )
        return _load_row(value)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_env": None}

    def close(self) -> None:
        """Release the environment; the next ``get`` reopens it. LMDB allows one
        open environment per path per process, so close before opening another."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        self.close()


def shard_path(root: str, shard: int) -> str:
    return os.path.join(root, SHARDS_DIR, f"{shard:06d}.tar")


class PackedStore:
    """format=packed: ``seek(offset)`` + ``read(size)`` inside ``shards/<shard>.tar``.

    ``loc`` offsets point at the member *data*, not the tar header, so a read is a
    single seek. Handles are raw ``io.FileIO`` objects (every read is an exact,
    whole-row slice, so buffering adds nothing), cached per process and evicted
    LRU at ``MAX_OPEN_SHARDS``.
    """

    def __init__(self, path: str, index: Index):
        self.path = path
        self.index = index
        self._handles: OrderedDict[int, io.FileIO] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _handle(self, shard: int) -> io.FileIO:
        handle = self._handles.get(shard)
        if handle is not None:
            self._handles.move_to_end(shard)
            return handle
        if len(self._handles) >= MAX_OPEN_SHARDS:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        handle = io.FileIO(shard_path(self.path, shard), "rb")
        self._handles[shard] = handle
        return handle

    def get(self, row_id: int) -> Row:
        entry = self.index.entries[row_id]
        shard, offset, size = cast(tuple[int, int, int], entry.loc)
        handle = self._handle(shard)
        handle.seek(offset)
        buffer = bytearray(size)
        view = memoryview(buffer)
        filled = 0
        while filled < size:
            # A raw file may return fewer bytes than asked for; loop until full.
            n = handle.readinto(view[filled:])
            if not n:
                raise OSError(
                    f"Short read for key {entry.key!r} in shard {shard} of {self.path}: "
                    f"expected {size} bytes at offset {offset}, got {filled}"
                )
            filled += n
        return _load_row(view)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_handles": OrderedDict()}

    def close(self) -> None:
        """Close every cached shard handle; the next ``get`` reopens on demand."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


class OnlineStore:
    """No cache: rows come straight from a (coerced) raw source, un-preprocessed.

    The index carries positional keys only (``cost=0``, ``sig=""``): reading the
    real ``key`` of every row up front would load every image / tensor.
    """

    def __init__(self, source: RawSource):
        self.source = source
        self.index = Index(
            [IndexEntry(str(i), 0, "", None, str(i)) for i in range(len(source))],
            {"format": "online"},
        )

    def __len__(self) -> int:
        return len(self.index)

    def get(self, row_id: int) -> Row:
        return dict(self.source[row_id])


def open_cache(path: str, *, limit: int | None = None) -> RowStore:
    """Open any cache directory by its ``meta.json``; ``limit`` truncates the index."""
    index = Index.load(path)
    if limit is not None and limit > 0:
        index = Index(index.entries[:limit], index.meta)
    fmt = index.meta.get("format")
    if fmt == "packed":
        return PackedStore(path, index)
    if fmt == "random":
        backend = index.meta.get("backend")
        if backend == "directory":
            return DirectoryStore(path, index)
        if backend == "lmdb":
            return LmdbStore(path, index)
        raise ValueError(f"{path}: unknown random cache backend {backend!r}")
    raise ValueError(f"{path}: unknown cache format {fmt!r}")
