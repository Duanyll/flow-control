import io
import uuid
import zlib
from typing import Any

import lmdb
import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr
from torch.utils.data import Dataset

from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import DataSink

logger = get_logger(__name__)

LMDB_MAX_DBS = 16


class LMDBDataset(BaseModel, Dataset):
    """LMDB-backed dataset. Pickle-safe: only config is serialized,
    the LMDB connection is lazily opened on first access."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    db_name: str | None = None
    max_items: int = 0
    use_compression: bool = False

    _env: lmdb.Environment | None = PrivateAttr(default=None)
    _db: Any = PrivateAttr(default=None)
    _keys: list[bytes] | None = PrivateAttr(default=None)

    def _ensure_open(self) -> None:
        if self._env is not None:
            return
        env = lmdb.open(
            self.path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=LMDB_MAX_DBS,
        )
        self._env = env
        self._db = env.open_db(self.db_name.encode()) if self.db_name else env.open_db()
        self._keys = self._load_keys()
        logger.info(
            f"LMDBDataset initialized with {len(self._keys)} items from {self.path} (db: {self.db_name})"
        )

    def _load_keys(self) -> list[bytes]:
        assert self._env is not None and self._db is not None
        with self._env.begin() as txn:
            cursor = txn.cursor(self._db)
            keys = []
            for i, key in enumerate(cursor.iternext(values=False)):
                if self.max_items > 0 and i >= self.max_items:
                    break
                keys.append(key)
            return keys

    def __len__(self) -> int:
        self._ensure_open()
        assert self._keys is not None
        return len(self._keys)

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        assert self._keys is not None and self._env is not None and self._db is not None
        if idx >= len(self._keys):
            raise IndexError(f"Index {idx} out of range.")
        key = self._keys[idx]
        with self._env.begin(write=False) as txn:
            value = txn.get(key, db=self._db)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            if self.use_compression:
                value = zlib.decompress(value)
            buffer = io.BytesIO(value)
            sample = torch.load(buffer)
        if "__key__" not in sample:
            sample["__key__"] = key.decode()
        return sample

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()

    def __getstate__(self) -> dict:
        return self.model_dump()

    def __setstate__(self, state: dict) -> None:
        new = type(self).model_validate(state)
        object.__setattr__(self, "__dict__", new.__dict__)
        object.__setattr__(self, "__pydantic_fields_set__", new.__pydantic_fields_set__)
        object.__setattr__(self, "__pydantic_extra__", new.__pydantic_extra__)
        object.__setattr__(self, "__pydantic_private__", new.__pydantic_private__)


class LMDBDataSink(DataSink):
    def setup(
        self,
        worker_id: int,
        path: str,
        db_name: str | None = None,
        map_size: int = 1 << 40,
        use_compression: bool = False,
    ):
        self.lmdb_path = path
        self.env = lmdb.open(
            path,
            map_size=map_size,
            max_dbs=LMDB_MAX_DBS,
        )
        self.db_name = db_name
        self.db = self.env.open_db(db_name.encode()) if db_name else self.env.open_db()
        self.use_compression = use_compression
        logger.info(
            f"LMDBDataSink initialized at {path} (db: {db_name}) for worker {worker_id}"
        )

    def write(self, item: dict):
        key = item["__key__"].encode() if "__key__" in item else uuid.uuid4().bytes
        with self.env.begin(write=True, db=self.db) as txn:
            buffer = io.BytesIO()
            torch.save(item, buffer)
            if self.use_compression:
                buffer = zlib.compress(buffer.getvalue())
            else:
                buffer = buffer.getvalue()
            txn.put(key, buffer)

        return True

    def cleanup(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
            logger.info(f"LMDBDataSink at {self.lmdb_path} closed.")
