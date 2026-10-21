from typing import Any, cast

import torch
import torch.distributed as dist
from pydantic import ConfigDict
from rich.progress import Progress
from rich.table import Table
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.data import (
    COST,
    IMAGE_SIZE,
    KEY,
    ReportConfig,
    ReportWriter,
    RowStream,
    build_loader,
    is_padding,
)
from flow_control.processors.base import DecodedRow
from flow_control.rewards import Scorers
from flow_control.rewards.base import load_rewards
from flow_control.samplers import Row, Trajectory, derive_seed
from flow_control.training.rollouts import SampleContext
from flow_control.utils.logging import console, dump_if_failed, get_logger
from flow_control.utils.tensor import deep_cast_float_dtype

from .mixins import (
    BaseTrainer,
    DcpMixin,
    LoggingMixin,
    RunnerMixin,
    distributed_main,
    score_one,
    trainer_registry,
)
from .mixins.base import load_model_state_dict, model_state_dict

logger = get_logger(__name__)


@trainer_registry.register("inference")
class Inference(RunnerMixin, BaseTrainer, DcpMixin):
    model_config = ConfigDict(extra="forbid")

    model: ModelAdapter
    sampler: Trajectory
    report: ReportConfig
    """Where the outputs go (``docs/data/formats.md``): ``metrics.jsonl`` with one line per
    sample, ``previews/<key>.png`` and optionally ``records/`` (a random cache of
    the full result rows)."""
    reward: Scorers | None = None
    """Scorers on every sample; their raw and normalized components go to
    ``metrics.jsonl`` and the records."""

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    variant: str | None = None
    """Model variant for the sampler's model leaves that name none (``docs/adapters/variants.md``)."""
    save_extra: bool = False
    """``records/`` rows hold the whole working row (source fields, conditions,
    decoded outputs) instead of only the decoded outputs plus key / cost /
    image_size."""
    annotate_output_image: bool = False
    """Save the processor's annotated preview image (e.g. tie source + edit, or layered
    layers merged with labels) as the preview instead of the bare clean image.
    Records and rewards always see the clean ``clean_image``."""

    # ------------------------------- Lazy state --------------------------------- #
    _stream: RowStream | None = None
    _dataloader: StatefulDataLoader | None = None

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def dataloader(self) -> StatefulDataLoader:
        if self._dataloader is None:
            raise RuntimeError("Dataloader not created yet.")
        return self._dataloader

    def make_dataloader(self):
        self._store = self.open_inference_store(self.dataset)
        self._stream = RowStream(
            self._store,
            self.make_planner(self._store, shuffle=False),
            self.rank,
            self.world_size,
        )
        self._dataloader = build_loader(
            self._stream, batch_size=1, num_workers=self.num_dataloader_workers
        )

    # ------------------------------- Checkpointing ------------------------------ #

    def state_dict(self):
        return {"transformer": model_state_dict(self.model)}

    def load_state_dict(self, state_dict: dict[str, Any]):
        load_model_state_dict(self.model, state_dict["transformer"])

    # ---------------------------------- Output ---------------------------------- #

    def _reward_fields(self, row: Row) -> dict[str, Any]:
        assert self.reward is not None
        raw, normalized = row["reward_raw"], row["reward"]
        weights = torch.tensor(self.reward.weights, dtype=normalized.dtype)
        labels = self.reward.labels
        return {
            "reward": (normalized * weights).sum(),
            "reward_raw": {label: raw[i].clone() for i, label in enumerate(labels)},
            "reward_normalized": {
                label: normalized[i].clone() for i, label in enumerate(labels)
            },
        }

    def _write_output(
        self, writer: ReportWriter, row: Row, decoded_keys: set[str]
    ) -> None:
        """One finished sample into the report; its scores go to ``metrics.jsonl``
        and, so that ``records/`` can be filtered by reward, into the record too.
        The row is on CPU already: the runner moved it there at decode."""
        preview = None
        if self.report.previews:
            preview = (
                self.processor.annotate_output(cast(DecodedRow, row), row)
                if self.annotate_output_image
                else row["clean_image"]
            )
        record = (
            dict(row)
            if self.save_extra
            else {
                KEY: row[KEY],
                COST: row[COST],
                IMAGE_SIZE: row[IMAGE_SIZE],
                **{key: row[key] for key in decoded_keys},
            }
        )
        metrics = self._reward_fields(row) if self.reward is not None else {}
        record.update(metrics)
        writer.write(record, preview, metrics)

    # ---------------------------------- Rewards --------------------------------- #

    def _gather_scored(
        self, scored: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Gather per-sample ``(raw, normalized)`` scores onto the main process
        (None elsewhere)."""
        if self.world_size <= 1:
            return scored
        if self.is_main_process:
            gathered: list[Any] = [None] * self.world_size
            dist.gather_object(scored, gathered, dst=0)
            return [item for part in gathered for item in part]
        dist.gather_object(scored, None, dst=0)
        return None

    def _print_reward_summary(
        self, scored: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        assert self.reward is not None
        raw = torch.stack([r for r, _ in scored], dim=0)
        normalized = torch.stack([n for _, n in scored], dim=0)
        weights = torch.tensor(self.reward.weights, dtype=normalized.dtype)
        aggregate = (normalized * weights).sum(dim=-1)

        console.rule("[bold green]Reward Summary[/bold green]")
        table = Table(
            caption=f"n={aggregate.shape[0]} samples over {self.world_size} rank(s)",
            header_style="bold",
        )
        table.add_column("component")
        for col in ("raw mean", "raw std", "norm mean", "norm std"):
            table.add_column(col, justify="right")
        table.add_row(
            "[bold]weighted[/bold]",
            "—",
            "—",
            f"{aggregate.mean().item():.4f}",
            f"{aggregate.std(correction=0).item():.4f}",
        )
        for i, label in enumerate(self.reward.labels):
            table.add_row(
                label,
                f"{raw[:, i].mean().item():.4f}",
                f"{raw[:, i].std(correction=0).item():.4f}",
                f"{normalized[:, i].mean().item():.4f}",
                f"{normalized[:, i].std(correction=0).item():.4f}",
            )
        console.print(table)

    def _report_rewards(self, scored: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Gather the ``(raw, normalized)`` scores across ranks and print the
        summary (the per-sample breakdown is already in ``metrics.jsonl``)."""
        all_scored = self._gather_scored(scored)
        if all_scored:
            self._print_reward_summary(all_scored)

    # ------------------------------- Main loop ---------------------------------- #

    @torch.no_grad()
    @distributed_main
    def run(self):
        if self.variant is not None:
            self.model.resolve(self.variant)
        self.set_seed()
        # Dump the run description before any weights load (as init_tracker does).
        meta = {"config": self.model_dump(mode="json", warnings="none")}
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.load_processor()
        self.make_runner(self.model)
        self.make_dataloader()
        if self.reward is not None:
            load_rewards(self.reward, self.device)

        if self.checkpoint_dir is not None:
            self.load_dcp_checkpoint(self.checkpoint_dir)

        writer = self.report.open(self.rank)
        logger.info(f"Writing the report to {self.report.path}")

        self.transformer.eval()
        console.rule("[bold green]Starting Inference[/bold green]")

        progress = Progress(
            *LoggingMixin.get_progress_columns(),
            console=console,
        )
        assert self._stream is not None
        task = progress.add_task("Inference", total=len(self._stream))

        def prepare(item: Row) -> Row:
            with dump_if_failed(logger, item):
                return deep_cast_float_dtype(
                    self.prepare_row(item, mode="inference", epoch=0),
                    self.model.dtype,
                )

        decoded_keys: dict[int, set[str]] = {}
        """Per row (by identity; the runner keeps the dict object): the keys
        ``decode_output`` wrote, which the record keeps."""

        async def sample(ctx: SampleContext) -> list[Row]:
            row = await ctx.start(derive_seed(self.seed, ctx.prompt[KEY]))
            await ctx.admit(self.sampler(row))
            decoded_keys[id(row)] = set(await ctx.decode(row))
            return [row]

        scored: list[tuple[torch.Tensor, torch.Tensor]] = []
        with self.profile_scope("inference"), progress:
            items = enumerate(item for batch in self.dataloader for item in batch)
            pending = self.runner.submit(
                items,
                sample,
                score_one(self.reward),
                prepare=prepare,
                k=1,
                epoch=0,
                variant=self.variant,
            )
            for (row,) in pending.as_completed():
                progress.advance(task)
                decoded = decoded_keys.pop(id(row))
                if is_padding(row):
                    continue
                self._write_output(writer, row, decoded)
                if self.reward is not None:
                    scored.append((row["reward_raw"], row["reward"]))

        # Collective: every rank closes its part, rank 0 merges.
        writer.finalize(meta=meta)
        if self.reward is not None:
            self._report_rewards(scored)
        console.rule("[bold green]Inference Completed[/bold green]")
