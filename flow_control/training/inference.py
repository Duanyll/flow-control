from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from pydantic import ConfigDict, model_validator
from rich.progress import Progress, TaskID
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
from flow_control.rewards import Reward, _has_pairwise_child, execute_reward
from flow_control.rewards.base import RewardResult
from flow_control.samplers import (
    Calls,
    Executor,
    Row,
    Trajectory,
    data,
    derive_seed,
    variants,
)
from flow_control.utils.logging import console, dump_if_failed, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
)

from .mixins import (
    BaseTrainer,
    DataMixin,
    DcpMixin,
    LoggingMixin,
    distributed_main,
    trainer_registry,
)
from .mixins.base import load_model_state_dict, model_state_dict

logger = get_logger(__name__)


@dataclass(slots=True)
class _Output:
    """One finished sample waiting for its score: the CPU record that goes to
    the report plus the preview image (None when previews are off)."""

    record: dict[str, Any]
    preview: torch.Tensor | None


@trainer_registry.register("inference")
class Inference(DataMixin, BaseTrainer, DcpMixin):
    model_config = ConfigDict(extra="forbid")

    model: ModelAdapter
    sampler: Trajectory
    report: ReportConfig
    """Where the outputs go (design §10): ``metrics.jsonl`` with one line per
    sample, ``previews/<key>.png`` and optionally ``records/`` (a random cache of
    the full result rows)."""
    reward: Reward | None = None

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    variant: str | None = None
    """Model variant for the sampler's model leaves that name none (``docs/model-variants.md``)."""
    save_extra: bool = False
    """``records/`` rows hold the whole working row (source fields, conditions,
    decoded outputs) instead of only the decoded outputs plus key / cost /
    image_size."""
    annotate_output_image: bool = False
    """Save the processor's annotated preview image (e.g. tie source + edit, or layered
    layers merged with labels) as the preview instead of the bare clean image.
    Records and rewards always see the clean ``clean_image``."""

    @model_validator(mode="after")
    def check_reward(self):
        if self.reward is not None and _has_pairwise_child(self.reward):
            raise ValueError(
                "A pairwise reward compares the rollouts of one prompt and cannot "
                "score inference rows one at a time; use a pointwise reward that "
                "compares each sample against an anchor (for example "
                "reference_compare in anchored mode)."
            )
        return self

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

    # ---------------------------------- Sampling -------------------------------- #

    def _runs(self) -> Iterator[tuple[Row, Calls[None]]]:
        """Prepare one row at a time as the executor asks for it."""
        for items in self.dataloader:
            for item in items:
                with dump_if_failed(logger, item):
                    row = deep_cast_float_dtype(
                        self.prepare_row(item, mode="inference", epoch=0),
                        self.model.dtype,
                    )
                    generator = torch.Generator(device=self.device).manual_seed(
                        derive_seed(self.seed, row[KEY])
                    )
                    row = self.start_row(row, generator, self.model.dtype)
                yield row, self.sampler(row)

    def _sample_submitter(
        self,
        progress: Progress,
        task: TaskID,
    ) -> Generator[tuple[dict[str, Any], _Output]]:
        """Sample, decode and yield ``(row, output)`` for scoring.

        ``row`` (on device) is handed to ``execute_reward``, which snapshots
        the fields it needs for async scoring and lets sampling of later
        requests overlap with the reward request still in flight. ``output`` is
        the CPU payload written to the report once its score is known.

        Padding rows still run the model (to keep FSDP collectives balanced
        across ranks) but are not yielded for scoring or output.
        """
        cpu = torch.device("cpu")
        executor = Executor(self.model, variants(self.sampler))
        for row in executor.stream(self._runs()):
            decoded = self.processor.decode_output(row["noisy_latents"], row)
            row.update(decoded)
            progress.advance(task)
            if is_padding(row):
                continue
            # Build the annotated preview while the full GPU row (e.g. tie's
            # reference_images) and decoded outputs are still available.
            preview = None
            if self.report.previews:
                preview = (
                    self.processor.annotate_output(decoded, row)
                    if self.annotate_output_image
                    else decoded["clean_image"]
                ).to(cpu)
            # The runtime keys (call chain, RNG, caches) are not picklable.
            record = deep_move_to_device(
                data(row)
                if self.save_extra
                else {
                    KEY: row[KEY],
                    COST: row[COST],
                    IMAGE_SIZE: row[IMAGE_SIZE],
                    **decoded,
                },
                cpu,
            )
            yield row, _Output(record, preview)

    # ---------------------------------- Output ---------------------------------- #

    @staticmethod
    def _reward_fields(result: RewardResult) -> dict[str, Any]:
        raw = result.raw.detach().cpu()
        normalized = result.normalized.detach().cpu()
        return {
            "reward": result.aggregate().detach().cpu(),
            "reward_raw": {
                label: raw[:, i].clone() for i, label in enumerate(result.labels)
            },
            "reward_normalized": {
                label: normalized[:, i].clone() for i, label in enumerate(result.labels)
            },
        }

    def _write_output(
        self, writer: ReportWriter, output: _Output, result: RewardResult | None
    ) -> None:
        """One sample into the report; its scores go to ``metrics.jsonl`` and,
        so that ``records/`` can be filtered by reward, into the record too."""
        metrics = self._reward_fields(result) if result is not None else {}
        output.record.update(metrics)
        writer.write(output.record, output.preview, metrics)

    # ---------------------------------- Rewards --------------------------------- #

    def _gather_scored(self, scored: list[RewardResult]) -> list[RewardResult] | None:
        """Gather per-sample scores onto the main process (None elsewhere)."""
        if self.world_size <= 1:
            return scored
        if self.is_main_process:
            gathered: list[Any] = [None] * self.world_size
            dist.gather_object(scored, gathered, dst=0)
            return [item for part in gathered for item in part]
        dist.gather_object(scored, None, dst=0)
        return None

    def _print_reward_summary(self, scored: list[RewardResult]) -> None:
        normalized = torch.cat([r.normalized for r in scored], dim=0)
        raw = torch.cat([r.raw for r in scored], dim=0)
        weights = scored[0].weights.to(dtype=normalized.dtype)
        labels = scored[0].labels
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
        for i, label in enumerate(labels):
            table.add_row(
                label,
                f"{raw[:, i].mean().item():.4f}",
                f"{raw[:, i].std(correction=0).item():.4f}",
                f"{normalized[:, i].mean().item():.4f}",
                f"{normalized[:, i].std(correction=0).item():.4f}",
            )
        console.print(table)

    def _report_rewards(self, scored: list[RewardResult]) -> None:
        """Gather scores across ranks and print the summary (the per-sample
        breakdown is already in the report's ``metrics.jsonl``)."""
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
        self.make_dataloader()
        if self.reward is not None:
            self.reward.load_model(self.device)

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

        with (
            self.profile_scope("inference"),
            progress,
            self.model.use_variant(self.variant),
        ):
            submitter = self._sample_submitter(progress, task)
            if self.reward is not None:

                def handler(output: _Output, result: RewardResult) -> RewardResult:
                    cpu_result = deep_move_to_device(result, torch.device("cpu"))
                    self._write_output(writer, output, cpu_result)
                    return cpu_result

                scored = execute_reward(self.reward, submitter, handler)
            else:
                for _row, output in submitter:
                    self._write_output(writer, output, None)
                scored = []

        # Collective: every rank closes its part, rank 0 merges.
        writer.finalize(meta=meta)
        if self.reward is not None:
            self._report_rewards(scored)
        console.rule("[bold green]Inference Completed[/bold green]")
