import lmdb
from torch.utils.data import Dataset
import pickle

from flow_control.utils.common import unpack_bool_tensor
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

class LMDBDataset(Dataset):
    def __init__(
        self,
        path: str,
        db_name: str | None = None,
        max_items: int = 0,
        unpack_bool=True,
    ):
        self.lmdb_path = path
        self.env = lmdb.open(
            path, readonly=True, lock=False, readahead=False, meminit=False, max_dbs=16
        )
        self.db_name = db_name
        self.db = self.env.open_db(db_name.encode()) if db_name else self.env.open_db()
        self.keys = self._load_keys()
        if len(self.keys) > max_items > 0:
            self.keys = self.keys[:max_items]
        logger.info(
            f"LMDBDataset initialized with {len(self.keys)} items from {path} (db: {db_name})"
        )
        self.unpack_bool = unpack_bool

    def _load_keys(self):
        with self.env.begin() as txn:
            return list(txn.cursor(self.db).iternext(values=False))

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
            sample = pickle.loads(value)

        if self.unpack_bool:
            sample = {
                k: unpack_bool_tensor(*v) if isinstance(v, tuple) and len(v) == 2 else v
                for k, v in sample.items()
            }
        return sample

    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
