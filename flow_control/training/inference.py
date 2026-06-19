import csv
import os
from collections.abc import Generator
from typing import Any

import torch
import torch.distributed as dist
from pydantic import ConfigDict, model_validator
from rich.progress import Progress, TaskID
from rich.table import Table
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.datasets import DatasetConfig, DatasinkConfig, parse_datasink
from flow_control.processors import Processor
from flow_control.rewards import Reward, execute_reward
from flow_control.rewards.base import RewardResult
from flow_control.samplers import Sampler
from flow_control.samplers.sampler import derive_seed
from flow_control.utils.logging import console, dump_if_failed, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
    tensor_to_pil,
)

from .data import (
    DistributedBucketSampler,
    PaddingAwareDatasetWrapper,
    collate_fn,
    seed_worker,
)
from .launch_config import trainer_registry
from .mixins import (
    BaseTrainer,
    DcpMixin,
    LoggingMixin,
    PreprocessMixin,
    distributed_main,
)

logger = get_logger(__name__)


@trainer_registry.register("inference")
class Inference(PreprocessMixin, BaseTrainer, DcpMixin):
    model_config = ConfigDict(extra="forbid")

    model: ModelAdapter
    sampler: Sampler
    processor: Processor
    dataset: DatasetConfig
    datasink: DatasinkConfig | None = None
    reward: Reward | None = None

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    save_preview_dir: str | None = None
    save_extra: bool = False
    annotate_output_image: bool = False
    """Save the processor's annotated preview image (e.g. tie source + edit, or layered
    layers merged with labels) to ``save_preview_dir`` instead of the bare clean image.
    The datasink always receives the clean ``clean_image``."""
    reward_csv_path: str | None = None
    """Optional path to write per-sample reward scores as a CSV.

    The terminal only shows the aggregated summary; set this to keep the
    full per-image breakdown. Written by the main process after gathering
    scores from all ranks.
    """

    @model_validator(mode="after")
    def check_save_preview_dir(self):
        if self.datasink is None and self.save_preview_dir is None:
            raise ValueError("Either datasink or save_preview_dir must be specified.")
        return self

    # ------------------------------- Lazy state --------------------------------- #
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
        dataset = PaddingAwareDatasetWrapper(self.parse_inference_dataset(self.dataset))
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.seed,
            grad_acc_steps=1,
        )
        self._dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )

    # ------------------------------- Checkpointing ------------------------------ #

    def state_dict(self):
        transformer_state_dict, optimizer_state_dict = get_state_dict(
            self.transformer,
            optimizers=[],
            options=StateDictOptions(strict=False),
        )
        return {
            "transformer": transformer_state_dict,
            "optimizer": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(
            self.transformer,
            [],
            model_state_dict=state_dict["transformer"],
            optim_state_dict=state_dict["optimizer"],
            options=StateDictOptions(strict=False),
        )

    # ---------------------------------- Sampling -------------------------------- #

    def _sample_one(self, batch: Any) -> tuple[Any, Any, str]:
        """Run the sampler + decode for a single batch.

        Returns ``(batch, decoded, key)`` where ``batch`` (still on device) has
        the decoded outputs merged in, ``decoded`` is the decoded-only dict, and
        ``key`` is the sample key (``"__padding__"`` for padding samples).
        """
        batch = deep_move_to_device(batch, self.device)
        batch = self.preprocess_for_inference(batch, save_extra=True)
        batch = deep_cast_float_dtype(batch, self.model.dtype)
        negative_batch: Any = (
            self.processor.get_negative_batch(batch)
            if self.sampler.cfg_scale > 1.0
            else None
        )
        key = batch.get("__key__", "unknown")
        seed = derive_seed(self.seed, key)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.processor.initialize_latents(
            batch,
            generator=generator,
            device=self.device,
            dtype=self.model.dtype,
        )
        sample_output = self.sampler.sample(
            self.model,
            batch,
            negative_batch=negative_batch,
            generator=generator,
        )
        decoded = self.processor.decode_output(
            sample_output.final_latents,
            batch,
        )
        batch.update(decoded)
        return batch, decoded, key

    def _sample_submitter(
        self,
        progress: Progress,
        task: TaskID,
    ) -> Generator[tuple[dict[str, Any], tuple[dict[str, Any], str]]]:
        """Generate samples and yield ``(batch, (record, key))`` for scoring.

        ``batch`` (on device) is handed to ``execute_reward``, which snapshots
        the fields it needs for async scoring and lets sampling of the next
        batch overlap with the reward request still in flight. ``record`` is the
        CPU payload written to the datasink / preview once its score is known.

        Padding samples still run the model (to keep FSDP collectives balanced
        across ranks) but are not yielded for scoring or output.
        """
        for batch in self.dataloader:
            with dump_if_failed(logger, batch):
                batch, decoded, key = self._sample_one(batch)
                # Build the annotated preview while the full GPU batch (e.g. tie's
                # reference_images) and decoded outputs are still available.
                preview = (
                    self.processor.annotate_output(decoded, batch)
                    if (self.annotate_output_image and key != "__padding__")
                    else None
                )
                record = (
                    None
                    if key == "__padding__"
                    else deep_move_to_device(
                        batch if self.save_extra else decoded,
                        torch.device("cpu"),
                    )
                )
                if record is not None and preview is not None:
                    record["__preview_image__"] = preview.to(torch.device("cpu"))
            progress.advance(task)
            if record is not None:
                yield batch, (record, key)

    # ---------------------------------- Output ---------------------------------- #

    def _reward_fields(self, result: RewardResult) -> dict[str, Any]:
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

    def _write_record(
        self,
        record: dict[str, Any],
        key: str,
        datasink: Any,
        reward_result: RewardResult | None,
    ) -> None:
        """Write a single CPU ``record`` to the datasink and preview dir."""
        # Pop the preview before the datasink write so the sink only stores the clean
        # ``clean_image``; the annotated preview is for the saved PNG only.
        preview = record.pop("__preview_image__", None)
        if reward_result is not None:
            record.update(self._reward_fields(reward_result))
        if datasink is not None:
            datasink.write(record)
        if self.save_preview_dir is not None:
            image = tensor_to_pil(
                preview if preview is not None else record["clean_image"]
            )
            image.save(os.path.join(self.save_preview_dir, f"{key}.png"))

    # ---------------------------------- Rewards --------------------------------- #

    def _gather_scored(
        self,
        scored: list[tuple[str, RewardResult]],
    ) -> list[tuple[str, RewardResult]] | None:
        """Gather per-sample scores onto the main process (None elsewhere)."""
        if self.world_size <= 1:
            return scored
        if self.is_main_process:
            gathered: list[Any] = [None] * self.world_size
            dist.gather_object(scored, gathered, dst=0)
            return [item for part in gathered for item in part]
        dist.gather_object(scored, None, dst=0)
        return None

    def _write_reward_csv(
        self,
        scored: list[tuple[str, RewardResult]],
        path: str,
    ) -> None:
        labels = scored[0][1].labels
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["key", "reward"]
            for label in labels:
                header.extend([f"raw/{label}", f"normalized/{label}"])
            writer.writerow(header)
            for key, result in scored:
                raw = result.raw.flatten().tolist()
                normalized = result.normalized.flatten().tolist()
                row = [key, f"{result.aggregate().item():.6f}"]
                for r, n in zip(raw, normalized, strict=True):
                    row.extend([f"{r:.6f}", f"{n:.6f}"])
                writer.writerow(row)
        logger.info(f"Wrote per-sample reward scores to {path}")

    def _print_reward_summary(
        self,
        scored: list[tuple[str, RewardResult]],
    ) -> None:
        normalized = torch.cat([r.normalized for _, r in scored], dim=0)
        raw = torch.cat([r.raw for _, r in scored], dim=0)
        weights = scored[0][1].weights.to(dtype=normalized.dtype)
        labels = scored[0][1].labels
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

    def _report_rewards(self, scored: list[tuple[str, RewardResult]]) -> None:
        """Gather scores across ranks, then print the summary and write CSV."""
        all_scored = self._gather_scored(scored)
        if all_scored is None or not all_scored:
            return
        if self.reward_csv_path is not None:
            self._write_reward_csv(all_scored, self.reward_csv_path)
        self._print_reward_summary(all_scored)

    # ------------------------------- Main loop ---------------------------------- #

    @torch.no_grad()
    @distributed_main
    def run(self):
        self.set_seed()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.load_processor()
        self.make_dataloader()
        if self.reward is not None:
            self.reward.load_model(self.device)

        if self.checkpoint_dir is not None:
            self.load_dcp_checkpoint(self.checkpoint_dir)

        datasink = parse_datasink(self.datasink) if self.datasink is not None else None

        if self.save_preview_dir is not None:
            os.makedirs(self.save_preview_dir, exist_ok=True)
            logger.info(f"Saving preview images to {self.save_preview_dir}")

        self.transformer.eval()
        console.rule("[bold green]Starting Inference[/bold green]")

        progress = Progress(
            *LoggingMixin.get_progress_columns(),
            console=console,
        )
        task = progress.add_task("Inference", total=len(self.dataloader))

        with progress:
            submitter = self._sample_submitter(progress, task)
            if self.reward is not None:

                def handler(
                    tag: tuple[dict[str, Any], str],
                    result: RewardResult,
                ) -> tuple[str, RewardResult]:
                    record, key = tag
                    cpu_result = deep_move_to_device(result, torch.device("cpu"))
                    self._write_record(record, key, datasink, cpu_result)
                    return key, cpu_result

                scored = execute_reward(self.reward, submitter, handler)
            else:
                for _batch, (record, key) in submitter:
                    self._write_record(record, key, datasink, None)
                scored = []

        if self.reward is not None:
            self._report_rewards(scored)
        console.rule("[bold green]Inference Completed[/bold green]")
