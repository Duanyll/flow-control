"""``DataMixin``: the consumer side of the data stack (design §9.1).

Opens the row store behind a ``DatasetConfig``, picks the grouping plan for it
(§6.3) and turns a fetched row into a model-ready batch: move to device, run the
processor online when the dataset is a raw source, then ``processor.resample``.
Whether a dataset is preprocessed online is decided by the store type alone
(``OnlineStore``), so there is no ``enable_preprocess`` switch.
"""

import asyncio
from functools import partial
from typing import Literal, cast

import torch

from flow_control.adapters.base import Batch
from flow_control.data import (
    COST,
    KEY,
    PADDING,
    DatasetConfig,
    Group,
    Index,
    OnlineStore,
    PackedStore,
    Planner,
    Row,
    RowStore,
    groups_packed,
    groups_plain,
    groups_shuffled,
    groups_sorted,
    open_store,
)
from flow_control.processors import Processor
from flow_control.processors.base import ProcessedBatch
from flow_control.samplers import Sampler, SampleRequest, derive_seed
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import deep_move_to_device

from .base import BaseTrainer

logger = get_logger(__name__)


def _sorted_planner(index: Index, n: int, epoch: int) -> list[Group]:
    """``groups_sorted`` with the planner signature (module-level so it pickles)."""
    return groups_sorted(index, n)


class DataMixin(BaseTrainer):
    processor: Processor
    dataset: DatasetConfig
    group_size: int = 16
    """Rows per plan group (§6.1): ``world_size`` must divide it and, for trainers
    with microbatches, ``train_micro_batch_size`` must divide ``group_size /
    world_size`` so one microbatch never straddles two groups."""
    megabatch_groups: int = 64
    """random cache: groups per megabatch; rows are cost-sorted inside a megabatch
    and megabatches are shuffled against each other."""
    shuffle_mode: Literal["auto", "strict", "shard"] = "auto"
    """packed cache only. ``auto`` / ``shard``: shards are permuted and read
    sequentially (``groups_packed``); ``strict``: a full permutation as on a
    random cache (``groups_shuffled``), at the price of random seeks."""
    num_dataloader_workers: int = 1

    _store: RowStore | None = None
    """The consumer's main store (training set / inference set / rollout prompts)."""
    _loop: asyncio.AbstractEventLoop | None = None
    """Event loop for the processor's async ``prepare_*_batch``; created together
    with the encode models the first time a raw-source store is opened."""

    @property
    def online_preprocess(self) -> bool:
        """True when ``_store`` is a raw source, i.e. rows still need the processor."""
        return isinstance(self._store, OnlineStore)

    def load_processor(self) -> None:
        """Decode models on every rank; encode models only once a raw-source store
        is opened (see ``_open_store``)."""
        self.processor.load_models("decode", self.device)

    def _open_store(
        self, config: DatasetConfig, mode: Literal["training", "inference"]
    ) -> RowStore:
        store = open_store(config, processor_class=type(self.processor), mode=mode)
        if isinstance(store, OnlineStore) and self._loop is None:
            logger.info(
                f"Dataset {config.get('type')!r} is a raw source: rows are "
                "preprocessed online, loading the processor's encode models."
            )
            self.processor.load_models("encode", self.device)
            self._loop = asyncio.new_event_loop()
        logger.info(f"Opened {type(store).__name__} with {len(store)} rows.")
        return store

    def open_train_store(self) -> RowStore:
        return self._open_store(self.dataset, "training")

    def open_inference_store(self, config: DatasetConfig) -> RowStore:
        return self._open_store(config, "inference")

    def make_planner(
        self, store: RowStore, *, shuffle: bool, micro_batch_size: int = 1
    ) -> Planner:
        """Pick the ``groups_*`` plan for ``store`` (§6.3) after checking that the
        ranks and microbatches tile ``group_size``. ``micro_batch_size`` is passed
        by trainers that own ``train_micro_batch_size``."""
        n = self.group_size
        if n % self.world_size != 0:
            raise ValueError(
                f"group_size ({n}) must be divisible by world_size "
                f"({self.world_size}); every rank takes group_size / world_size rows "
                "of each group."
            )
        per_rank = n // self.world_size
        if per_rank % micro_batch_size != 0:
            raise ValueError(
                f"train_micro_batch_size ({micro_batch_size}) must divide "
                f"group_size / world_size ({per_rank}); otherwise one microbatch "
                "spans two groups of different cost."
            )
        if isinstance(store, OnlineStore):
            return partial(groups_plain, len(store), n, self.seed, shuffle=shuffle)
        if not shuffle:
            return partial(_sorted_planner, store.index, n)
        if isinstance(store, PackedStore) and self.shuffle_mode != "strict":
            return partial(groups_packed, store.index, n, self.seed)
        if self.shuffle_mode == "shard":
            raise ValueError(
                "shuffle_mode='shard' needs a packed cache (`flow-control pack`); "
                f"{type(store).__name__} has no shards. Use 'auto' or 'strict'."
            )
        return partial(
            groups_shuffled, store.index, n, self.megabatch_groups, self.seed
        )

    def _preprocess_online(
        self, row: Row, mode: Literal["training", "inference"]
    ) -> Row:
        if self._loop is None:
            raise RuntimeError(
                "Online preprocessing requested but no raw-source store was opened "
                "through open_train_store / open_inference_store."
            )
        processed: Row = dict(
            self._loop.run_until_complete(
                self.processor.prepare_training_batch(row)
                if mode == "training"
                else self.processor.prepare_inference_batch(row)
            )
        )
        if mode == "inference":
            # Keep the raw fields (prompt captions, save_extra records); the
            # processor's rewritten values win.
            processed = {**row, **processed}
        for name in (KEY, PADDING):
            if name in row:
                processed[name] = row[name]
        processed[COST] = self.processor.get_cost(cast(ProcessedBatch, processed))
        return processed

    def prepare_row(
        self,
        row: Row,
        *,
        mode: Literal["training", "inference"],
        epoch: int,
        online: bool | None = None,
    ) -> ProcessedBatch:
        """Fetched row -> batch on ``self.device``: online preprocessing when the
        row's store is a raw source (``online`` defaults to ``online_preprocess``,
        i.e. the main store; pass it for rows of another store), then the
        processor's ``resample`` with a generator seeded by ``(epoch, key)`` so the
        draw does not depend on the dataloader worker that fetched the row."""
        row = deep_move_to_device(row, self.device)
        if self.online_preprocess if online is None else online:
            row = self._preprocess_online(row, mode)
        generator = torch.Generator(device=self.device).manual_seed(
            derive_seed(self.seed, f"{epoch}:{row[KEY]}")
        )
        return cast(ProcessedBatch, self.processor.resample(row, generator))

    def build_sample_request(
        self,
        sampler: Sampler,
        batch: Batch,
        generator: torch.Generator,
    ) -> SampleRequest:
        negative_batch = (
            self.processor.get_negative_batch(cast(ProcessedBatch, batch))
            if sampler.guidance.requires_negative(sampler.steps)
            else None
        )
        return SampleRequest(
            batch=batch,
            negative_batch=cast(Batch | None, negative_batch),
            generator=generator,
        )


if __name__ == "__main__":
    import tempfile
    from typing import Any
    from unittest.mock import patch

    from rich import print

    from flow_control.data import (
        RandomCacheWriter,
        RowStream,
        build_loader,
        finalize_cache,
    )
    from flow_control.processors import parse_processor
    from flow_control.processors.tasks.t2i import T2IProcessor

    class Probe(DataMixin):
        launch: Any = None

        @property
        def device(self):
            return torch.device("cpu")

    processor = parse_processor({"task": "t2i", "preset": "flux1"})
    with tempfile.TemporaryDirectory() as tmp:
        writer = RandomCacheWriter(0, tmp, "directory")
        for i in range(7):
            writer.write(
                {
                    KEY: f"k{i}",
                    COST: 10 + i,
                    "image_size": (32, 32),
                    "clean_latents": torch.stack(
                        [torch.full((4, 3), float(i)), torch.ones(4, 3)]
                    ),
                    "prompt_embeds": torch.zeros(1, 2, 3),
                }
            )
        writer.cleanup()
        finalize_cache(tmp, meta={"backend": "directory", "mode": "training"})

        probe = Probe(
            processor=processor, dataset={"type": "cache", "path": tmp}, group_size=4
        )
        store = probe.open_train_store()
        probe._store = store
        assert not probe.online_preprocess
        planner = probe.make_planner(store, shuffle=True, micro_batch_size=2)
        stream = RowStream(store, planner, 0, 1)
        loader = build_loader(stream, batch_size=2, num_workers=0)
        rows = [row for items in loader for row in items]
        assert len(rows) == 8 and sum(bool(r.get(PADDING)) for r in rows) == 1
        batch = probe.prepare_row(rows[0], mode="training", epoch=0)
        again = probe.prepare_row(
            store.get(int(rows[0][KEY][1:])), mode="training", epoch=0
        )
        other = probe.prepare_row(
            store.get(int(rows[0][KEY][1:])), mode="training", epoch=1
        )
        print(rows[0][KEY], batch["clean_latents"].shape)
        assert batch["clean_latents"].shape == (1, 4, 3)
        assert torch.equal(batch["clean_latents"], again["clean_latents"])
        assert not torch.equal(batch["clean_latents"], other["clean_latents"])

        for bad in ({"micro_batch_size": 3}, {"world_size": 3}):
            probe._world_size = bad.get("world_size", 1)
            try:
                probe.make_planner(
                    store, shuffle=True, micro_batch_size=bad.get("micro_batch_size", 1)
                )
            except ValueError as e:
                print("rejected:", e)
            else:
                raise AssertionError(bad)
        probe._world_size = 1
        probe.shuffle_mode = "shard"
        try:
            probe.make_planner(store, shuffle=True)
        except ValueError as e:
            print("rejected:", e)
        else:
            raise AssertionError("shard mode on a random cache")

    # No cache: the raw row goes through the processor on this rank, keeps its
    # key / padding flag and gets a cost.
    async def fake_prepare(self, batch):
        return {"image_size": (32, 32), "prompt_embeds": torch.zeros(1, 5, 3)}

    with (
        patch.object(T2IProcessor, "prepare_inference_batch", fake_prepare),
        patch.object(T2IProcessor, "load_models", lambda self, mode, device=None: None),
    ):
        online_probe = Probe(
            processor=processor,
            dataset={"type": "inline", "data": [{"prompt": "p"}, {"prompt": "q"}]},
        )
        online_store = online_probe.open_inference_store(online_probe.dataset)
        online_probe._store = online_store
        assert online_probe.online_preprocess
        row = online_store.get(1)
        row[PADDING] = True
        out: Any = online_probe.prepare_row(row, mode="inference", epoch=0)
        print(out)
        assert out[KEY] == "1" and out[PADDING] is True and out["prompt"] == "q"
        assert out[COST] == processor.get_cost(out)
