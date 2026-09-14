"""``index.jsonl`` + ``meta.json``: the row catalogue every cache format shares (design §4.1)."""

import json
import os
from dataclasses import dataclass
from typing import Any

INDEX_FILE = "index.jsonl"
META_FILE = "meta.json"


@dataclass(slots=True, frozen=True)
class IndexEntry:
    key: str
    cost: int
    sig: str
    """``shape_signature(row)``: sort tie-break after ``cost``."""
    image_size: tuple[int, int] | None
    loc: str | tuple[int, int, int]
    """directory: relative path; lmdb: key; packed: ``(shard, offset, size)``."""

    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "cost": self.cost,
                "sig": self.sig,
                "image_size": list(self.image_size) if self.image_size else None,
                "loc": list(self.loc) if isinstance(self.loc, tuple) else self.loc,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> "IndexEntry":
        d = json.loads(line)
        image_size = d["image_size"]
        loc = d["loc"]
        return cls(
            key=d["key"],
            cost=int(d["cost"]),
            sig=d["sig"],
            image_size=(int(image_size[0]), int(image_size[1])) if image_size else None,
            loc=(int(loc[0]), int(loc[1]), int(loc[2]))
            if isinstance(loc, list)
            else loc,
        )


def read_index_entries(path: str) -> list[IndexEntry]:
    """Parse one ``*.jsonl`` index file (the merged index or a per-worker part)."""
    with open(path, encoding="utf-8") as f:
        return [IndexEntry.from_json(line) for line in f if line.strip()]


def _write_index_entries(path: str, entries: list[IndexEntry]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_json())
            f.write("\n")


class Index:
    def __init__(self, entries: list[IndexEntry], meta: dict[str, Any]):
        self.entries = entries
        self.meta = meta

    @classmethod
    def load(cls, path: str) -> "Index":
        meta_path = os.path.join(path, META_FILE)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"{path} has no {META_FILE}, so it is not a flow_control cache. Old "
                "pickle_directory / bucket_directory / lmdb caches are not supported; "
                "re-run `flow-control preprocess` (and `flow-control pack` if needed)."
            )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return cls(read_index_entries(os.path.join(path, INDEX_FILE)), meta)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        _write_index_entries(os.path.join(path, INDEX_FILE), self.entries)
        with open(os.path.join(path, META_FILE), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def __len__(self) -> int:
        return len(self.entries)


if __name__ == "__main__":
    import tempfile

    from rich import print

    entries = [
        IndexEntry("a", 3, "x:1x2", (8, 8), "rows/aa/a.pt"),
        IndexEntry("b", 1, "", None, "b"),
        IndexEntry("c", 2, "y:[2]", (4, 8), (0, 512, 1024)),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        Index(entries, {"format": "random", "backend": "directory"}).save(tmp)
        loaded = Index.load(tmp)
        print(loaded.meta, loaded.entries)
        assert loaded.entries == entries
        assert loaded.meta["backend"] == "directory"
