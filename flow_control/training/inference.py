from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from pydantic import ConfigDict, model_validator
from rich.progress import Progress, TaskID
from rich.table import Table
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
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
from flow_control.processors import Processor
from flow_control.rewards import Reward, execute_reward
from flow_control.rewards.base import RewardResult
from flow_control.samplers import Sampler, SampleRequest, derive_seed
from flow_control.utils.logging import console, dump_if_failed, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
)

from .ema import EMAConfig, EMAOptimizer
from .mixins import (
    BaseTrainer,
    DataMixin,
    DcpMixin,
    LoggingMixin,
    distributed_main,
    trainer_registry,
)

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
    sampler: Sampler
    processor: Processor
    report: ReportConfig
    """Where the outputs go (design §10): ``metrics.jsonl`` with one line per
    sample, ``previews/<key>.png`` and optionally ``records/`` (a random cache of
    the full result rows)."""
    reward: Reward | None = None

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_weights: Literal["current", "ema", "ema_old"] = "current"
    """Which weights to apply from a training DCP checkpoint.

    ``ema`` selects the validation EMA stored as ``optim_ema`` and ``ema_old``
    selects the lagged rollout EMA stored as ``optim_ema_old``. The EMA options
    require a checkpoint produced by a trainer that saves the corresponding
    optimizer state.
    """
    save_extra: bool = False
    """``records/`` rows hold the whole working batch (source fields, conditions,
    decoded outputs) instead of only the decoded outputs plus key / cost /
    image_size."""
    annotate_output_image: bool = False
    """Save the processor's annotated preview image (e.g. tie source + edit, or layered
    layers merged with labels) as the preview instead of the bare clean image.
    Records and rewards always see the clean ``clean_image``."""

    @model_validator(mode="after")
    def check_checkpoint_weights(self):
        if self.checkpoint_weights != "current" and self.checkpoint_dir is None:
            raise ValueError(
                f"checkpoint_weights={self.checkpoint_weights!r} requires "
                "checkpoint_dir to be specified."
            )
        return self

    # ------------------------------- Lazy state --------------------------------- #
    _stream: RowStream | None = None
    _dataloader: StatefulDataLoader | None = None
    _checkpoint_ema_optimizer: EMAOptimizer | None = None

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

    def make_checkpoint_ema_optimizer(self) -> None:
        if self.checkpoint_weights == "current":
            return
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                f"checkpoint_weights={self.checkpoint_weights!r} requires trainable "
                "parameters, but none were found."
            )
        # The saved optimizer state supplies the EMA buffers. Its configured
        # decay is irrelevant because inference never steps this optimizer.
        self._checkpoint_ema_optimizer = EMAOptimizer(params, EMAConfig())

    def state_dict(self):
        opts = StateDictOptions(strict=False, ignore_frozen_params=True)
        state: dict[str, Any] = {
            "transformer": get_model_state_dict(self.transformer, options=opts),
        }
        if self._checkpoint_ema_optimizer is not None:
            key = "optim_ema" if self.checkpoint_weights == "ema" else "optim_ema_old"
            state[key] = get_optimizer_state_dict(
                self.transformer,
                self._checkpoint_ema_optimizer,
                options=opts,
            )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        opts = StateDictOptions(strict=False, ignore_frozen_params=True)
        set_model_state_dict(self.transformer, state_dict["transformer"], options=opts)
        if self._checkpoint_ema_optimizer is not None:
            key = "optim_ema" if self.checkpoint_weights == "ema" else "optim_ema_old"
            set_optimizer_state_dict(
                self.transformer,
                self._checkpoint_ema_optimizer,
                state_dict[key],
                options=opts,
            )
            self._checkpoint_ema_optimizer.coerce_buffer_dtype()

    # ---------------------------------- Sampling -------------------------------- #

    def _requests(self) -> Iterator[SampleRequest]:
        """Prepare one row at a time as the sampler asks for it."""
        for items in self.dataloader:
            for item in items:
                with dump_if_failed(logger, item):
                    batch = self.prepare_row(item, mode="inference", epoch=0)
                    batch = deep_cast_float_dtype(batch, self.model.dtype)
                    generator = torch.Generator(device=self.device).manual_seed(
                        derive_seed(self.seed, batch[KEY])
                    )
                    self.processor.initialize_latents(
                        batch,
                        generator=generator,
                        device=self.device,
                        dtype=self.model.dtype,
                    )
                    request = self.build_sample_request(self.sampler, batch, generator)
                yield request

    def _sample_submitter(
        self,
        progress: Progress,
        task: TaskID,
    ) -> Generator[tuple[dict[str, Any], _Output]]:
        """Sample, decode and yield ``(batch, output)`` for scoring.

        ``batch`` (on device) is handed to ``execute_reward``, which snapshots
        the fields it needs for async scoring and lets sampling of later
        requests overlap with the reward request still in flight. ``output`` is
        the CPU payload written to the report once its score is known.

        Padding rows still run the model (to keep FSDP collectives balanced
        across ranks) but are not yielded for scoring or output.
        """
        cpu = torch.device("cpu")
        for run in self.sampler.sample(self.model, self._requests()):
            batch: Any = run.batch
            decoded = self.processor.decode_output(run.ctx.latents, batch)
            batch.update(decoded)
            progress.advance(task)
            if is_padding(batch):
                continue
            # Build the annotated preview while the full GPU batch (e.g. tie's
            # reference_images) and decoded outputs are still available.
            preview = None
            if self.report.previews:
                preview = (
                    self.processor.annotate_output(decoded, batch)
                    if self.annotate_output_image
                    else decoded["clean_image"]
                ).to(cpu)
            record = deep_move_to_device(
                batch
                if self.save_extra
                else {
                    KEY: batch[KEY],
                    COST: batch[COST],
                    IMAGE_SIZE: batch[IMAGE_SIZE],
                    **decoded,
                },
                cpu,
            )
            yield batch, _Output(record, preview)

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
        self.set_seed()
        # Dump the run description before any weights load (as init_tracker does).
        meta = {"config": self.model_dump(mode="json", warnings="none")}
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.load_processor()
        self.make_dataloader()
        if self.reward is not None:
            self.reward.load_model(self.device)

        if self.checkpoint_dir is not None:
            self.make_checkpoint_ema_optimizer()
            self.load_dcp_checkpoint(self.checkpoint_dir)
            if self._checkpoint_ema_optimizer is not None:
                self._checkpoint_ema_optimizer.apply_shadow()
                logger.info(
                    f"Applied checkpoint {self.checkpoint_weights} weights for "
                    "inference."
                )

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

        with progress:
            submitter = self._sample_submitter(progress, task)
            if self.reward is not None:

                def handler(output: _Output, result: RewardResult) -> RewardResult:
                    cpu_result = deep_move_to_device(result, torch.device("cpu"))
                    self._write_output(writer, output, cpu_result)
                    return cpu_result

                scored = execute_reward(self.reward, submitter, handler)
            else:
                for _batch, output in submitter:
                    self._write_output(writer, output, None)
                scored = []

        # Collective: every rank closes its part, rank 0 merges.
        writer.finalize(meta=meta)
        if self.reward is not None:
            self._report_rewards(scored)
        console.rule("[bold green]Inference Completed[/bold green]")
