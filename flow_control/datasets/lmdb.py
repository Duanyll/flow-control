import io
import uuid
import zlib

import lmdb
import torch
from torch.utils.data import Dataset

from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import DataSink

logger = get_logger(__name__)

LMDB_MAX_DBS = 16


class LMDBDataset(Dataset):
    def __init__(
        self,
        path: str,
        db_name: str | None = None,
        max_items: int = 0,
        use_compression: bool = False,
    ):
        self.lmdb_path = path
        self.env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=LMDB_MAX_DBS,
        )
        self.db_name = db_name
        self.db = self.env.open_db(db_name.encode()) if db_name else self.env.open_db()
        self.keys = self._load_keys(max_items=max_items)
        logger.info(
            f"LMDBDataset initialized with {len(self.keys)} items from {path} (db: {db_name})"
        )
        self.use_compression = use_compression

    def _load_keys(self, max_items: int = 0):
        with self.env.begin() as txn:
            cursor = txn.cursor(self.db)
            keys = []
            for i, key in enumerate(cursor.iternext(values=False)):
                if max_items > 0 and i >= max_items:
                    break
                keys.append(key)
            return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range.")
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key, db=self.db)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            if self.use_compression:
                value = zlib.decompress(value)
            buffer = io.BytesIO(value)
            sample = torch.load(buffer)
        return sample

    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()


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
