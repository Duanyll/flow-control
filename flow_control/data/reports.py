"""Inference / evaluation reports (design §10).

Layout::

    <report>/
      meta.json          # caller-supplied run description + "rows", "ranks"
      metrics.jsonl      # one line per sample: key, rank, image_size, cost, <metrics...>
      previews/<key>.png # preview image handed to ``write``
      records/           # optional: full result rows as a random cache
                         # (rows/ + index.jsonl + meta.json, i.e. {"type": "cache"})

Every rank writes its own ``metrics-<rank>.jsonl`` (and, with ``records``, its
own index part through ``RandomCacheWriter``); ``finalize`` is a collective that
merges them on rank 0.
"""

import glob
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict

from flow_control.data.index import META_FILE
from flow_control.data.rows import COST, IMAGE_SIZE, KEY, Row, is_padding
from flow_control.data.writer import RandomCacheWriter, finalize_cache
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import tensor_to_pil

logger = get_logger(__name__)

METRICS_FILE = "metrics.jsonl"
PREVIEWS_DIR = "previews"
RECORDS_DIR = "records"


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    previews: bool = True
    """Save the preview image of every sample under previews/<key>.png."""
    records: bool = False
    """Also store the full result rows under records/ as a random cache that can
    be opened with {"type": "cache", "path": "<path>/records"}."""

    def open(self, rank: int) -> "ReportWriter":
        return ReportWriter(
            self.path, rank, previews=self.previews, records=self.records
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _flatten(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"reward_raw": {"a": x}}`` -> ``{"reward_raw.a": x}``."""
    flat: dict[str, Any] = {}
    for name, value in metrics.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{name}."))
        else:
            flat[f"{prefix}{name}"] = _jsonable(value)
    return flat


def _metrics_part(root: str, rank: int) -> str:
    return os.path.join(root, f"metrics-{rank}.jsonl")


class ReportWriter:
    def __init__(self, root: str, rank: int, *, previews: bool, records: bool):
        self.root = root
        self.rank = rank
        self.previews = previews
        os.makedirs(root, exist_ok=True)
        if previews:
            os.makedirs(os.path.join(root, PREVIEWS_DIR), exist_ok=True)
        # Truncate this rank's part: a stale one from an earlier run would be merged.
        open(_metrics_part(root, rank), "w", encoding="utf-8").close()
        self._records = (
            RandomCacheWriter(rank, os.path.join(root, RECORDS_DIR), "directory")
            if records
            else None
        )

    def write(
        self, row: Row, preview: torch.Tensor | None, metrics: dict[str, Any]
    ) -> None:
        """Append one sample. ``row`` must carry ``__key__`` (and ``cost`` when
        ``records`` is on: the records cache needs it like any other cache).
        Padding rows are dropped: they were forwarded only to keep ranks in step."""
        if is_padding(row):
            return
        key = row[KEY]
        line = {
            "key": key,
            "rank": self.rank,
            "image_size": _jsonable(row.get(IMAGE_SIZE)),
            "cost": row.get(COST),
            **_flatten(metrics),
        }
        with open(_metrics_part(self.root, self.rank), "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False))
            f.write("\n")
        if self.previews and preview is not None:
            tensor_to_pil(preview).save(
                os.path.join(self.root, PREVIEWS_DIR, f"{key}.png")
            )
        if self._records is not None:
            self._records.write(row)

    def finalize(self, *, meta: dict[str, Any] | None = None) -> None:
        """Collective: every rank closes its parts, then rank 0 merges
        ``metrics-<rank>.jsonl`` into ``metrics.jsonl``, writes ``meta.json``
        (``meta`` + row / rank counts) and finalizes ``records/`` as a cache."""
        if self._records is not None:
            self._records.cleanup()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.rank != 0:
            return

        parts = sorted(
            glob.glob(os.path.join(self.root, "metrics-*.jsonl")),
            key=lambda p: int(os.path.basename(p)[len("metrics-") : -len(".jsonl")]),
        )
        rows = 0
        with open(os.path.join(self.root, METRICS_FILE), "w", encoding="utf-8") as out:
            for part in parts:
                with open(part, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            rows += 1
        for part in parts:
            os.remove(part)
        with open(os.path.join(self.root, META_FILE), "w", encoding="utf-8") as f:
            json.dump({**(meta or {}), "rows": rows, "ranks": len(parts)}, f, indent=2)
            f.write("\n")
        if self._records is not None:
            finalize_cache(
                os.path.join(self.root, RECORDS_DIR), meta={"backend": "directory"}
            )
        logger.info(f"Report at {self.root}: {rows} rows from {len(parts)} ranks")


if __name__ == "__main__":
    import tempfile

    from rich import print

    from flow_control.data.store import open_cache

    with tempfile.TemporaryDirectory() as tmp:
        writers = [ReportWriter(tmp, r, previews=True, records=True) for r in range(2)]
        for i in range(5):
            row: Row = {
                KEY: f"k{i}",
                COST: 3 + i,
                IMAGE_SIZE: (8, 8),
                "clean_image": torch.rand(1, 3, 8, 8),
            }
            writers[i % 2].write(
                row,
                row["clean_image"],
                {
                    "reward": torch.tensor([0.5 * i]),
                    "reward_raw": {"a": torch.tensor([i])},
                },
            )
        writers[1].write({KEY: "pad", "__padding__": True}, None, {})
        # Without torch.distributed there is no barrier: close rank 1 before rank 0 merges.
        for w in reversed(writers):
            w.finalize(meta={"note": "smoke"})
        print(sorted(os.listdir(tmp)))
        with open(os.path.join(tmp, METRICS_FILE)) as f:
            lines = [json.loads(line) for line in f]
        print(lines)
        assert [line["key"] for line in lines] == ["k0", "k2", "k4", "k1", "k3"]
        assert lines[1] == {
            "key": "k2", "rank": 0, "image_size": [8, 8], "cost": 5,
            "reward": 1.0, "reward_raw.a": 2,
        }  # fmt: skip
        assert sorted(os.listdir(os.path.join(tmp, PREVIEWS_DIR))) == [
            f"k{i}.png" for i in range(5)
        ]
        with open(os.path.join(tmp, META_FILE)) as f:
            assert json.load(f) == {"note": "smoke", "rows": 5, "ranks": 2}
        records = open_cache(os.path.join(tmp, RECORDS_DIR))
        assert [e.key for e in records.index.entries] == [f"k{i}" for i in range(5)]
        assert torch.equal(records.get(0)["clean_image"], records.get(0)["clean_image"])
