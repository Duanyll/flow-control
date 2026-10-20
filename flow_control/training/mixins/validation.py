import time
from collections.abc import Awaitable, Callable
from typing import Literal, cast

import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.data import (
    KEY,
    DatasetConfig,
    OnlineStore,
    RowStore,
    RowStream,
    build_loader,
    is_padding,
)
from flow_control.processors.base import DecodedRow
from flow_control.rewards import Scorers
from flow_control.samplers import Row, Trajectory, derive_seed
from flow_control.training.rollouts import RewardContext, SampleContext, write
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import deep_cast_float_dtype

from .base import BaseTrainer
from .logging import LoggingMixin
from .runner import RunnerMixin

logger = get_logger(__name__)


def sample_one(
    sampler: Trajectory, base_seed: int
) -> Callable[[SampleContext], Awaitable[list[Row]]]:
    """The ``sample`` of a validation / inference group: one row per item,
    seeded by ``derive_seed(base_seed, key)``, sampled with *sampler* and decoded."""

    async def sample(ctx: SampleContext) -> list[Row]:
        row = await ctx.start(derive_seed(base_seed, ctx.prompt[KEY]))
        await ctx.admit(sampler(row))
        await ctx.decode(row)
        return [row]

    return sample


def score_one(
    scorer: Scorers | None,
) -> Callable[[RewardContext, list[Row]], Awaitable[list[Row]]]:
    """The matching ``score``: *scorer* on the row unless it is padding; the
    rows come back either way, since the main thread's image logging and the
    report are collective over every item."""

    async def score(ctx: RewardContext, rows: list[Row]) -> list[Row]:
        if scorer is not None and not is_padding(rows[0]):
            raw = await scorer.score(ctx, rows)
            write(rows, raw, scorer.normalize(raw))
        return rows

    return score


class ValidationMixin(RunnerMixin, LoggingMixin, BaseTrainer, BaseModel):
    """
    Mixin that provides validation: sampling images and optionally scoring rewards.
    """

    # ---------------------------------- Configs --------------------------------- #
    validation_dataset: DatasetConfig | None = None
    validation_num_workers: int = 1
    validation_same_seed: bool = True
    validation_log_images: bool | int = True
    validation_annotate_images: bool = True
    """Log the processor's annotated preview (e.g. tie source + edit, or layered layers,
    merged with labels) instead of the bare generated image. No-op for tasks whose
    ``annotate_output`` returns ``clean_image`` unchanged (plain T2I)."""
    validation_log_rewards: bool = True
    validation_scorer: Scorers | Literal[False] | None = None
    """Scorers of the validation metrics. ``null``: the RL trainer's
    ``rollout.reward`` (SFT scores nothing); ``false``: no scoring."""
    validation_at_start: bool = True
    """Validate before the first training step (the seed weights, or the resumed
    checkpoint's). ``false`` skips that pass, e.g. when relaunching a run whose
    step-0 numbers are already logged."""
    validation_variants: list[str] = ["default"]
    """Model variants to validate; the first logs as ``val``, the others as ``val/<name>``."""
    seed: int = 42

    validation_sampler: Trajectory

    # -------------------------------- Properties -------------------------------- #

    _validation_store: RowStore | None = None
    _validation_stream: RowStream | None = None
    _validation_dataloader: StatefulDataLoader | None = None

    @property
    def validation_dataloader(self) -> StatefulDataLoader:
        if self._validation_dataloader is None:
            raise ValueError("Validation dataloader not created yet.")
        return self._validation_dataloader

    # ---------------------------------- Methods ---------------------------------- #

    def make_validation_dataloader(
        self,
    ) -> None:
        """Create the validation dataloader (cost-sorted plan, never shuffled)."""
        if self.validation_dataset is None:
            logger.info("No validation dataset configured, skipping.")
            return

        store = self.open_inference_store(self.validation_dataset)
        self._validation_store = store
        self._validation_stream = RowStream(
            store, self.make_planner(store, shuffle=False), self.rank, self.world_size
        )
        self._validation_dataloader = build_loader(
            self._validation_stream,
            batch_size=1,
            num_workers=self.validation_num_workers,
        )
        logger.info(
            f"Validation dataloader created with {len(store)} samples "
            f"({len(self._validation_stream)} per rank)."
        )

    @staticmethod
    def _reward_metrics(
        rows: list[Row], scorer: Scorers, metric_prefix: str
    ) -> dict[str, float]:
        """Per-rank reward statistics; empty when this rank scored nothing (all
        its rows were padding), so ``log_metrics`` still runs as a collective."""
        if not rows:
            return {}
        normalized = torch.stack([row["reward"] for row in rows], dim=0)
        raw = torch.stack([row["reward_raw"] for row in rows], dim=0)
        weights = torch.tensor(scorer.weights, dtype=normalized.dtype)
        aggregate = (normalized * weights).sum(dim=-1)
        metrics: dict[str, float] = {
            f"{metric_prefix}/reward_mean": aggregate.mean().item(),
            f"{metric_prefix}/reward_std": aggregate.std(correction=0).item(),
        }
        for i, label in enumerate(scorer.labels):
            metrics[f"{metric_prefix}/raw/{label}_mean"] = raw[:, i].mean().item()
            metrics[f"{metric_prefix}/raw/{label}_std"] = (
                raw[:, i].std(correction=0).item()
            )
            metrics[f"{metric_prefix}/normalized/{label}_mean"] = (
                normalized[:, i].mean().item()
            )
            metrics[f"{metric_prefix}/normalized/{label}_std"] = (
                normalized[:, i].std(correction=0).item()
            )
        return metrics

    def validate_variants(
        self, model: ModelAdapter, step: int, default: Scorers | None
    ) -> None:
        """Validate once per ``validation_variants`` entry, scored by
        ``validation_scorer`` (``false``: none; ``null``: *default*)."""
        scorer = (
            None
            if self.validation_scorer is False
            else (self.validation_scorer or default)
        )
        for index, name in enumerate(self.validation_variants):
            suffix = f"/{name}" if index else ""
            self.validate_and_log(
                model,
                step,
                scorer,
                variant=name,
                metric_prefix=f"val{suffix}",
                image_name=f"validation{suffix}",
                profile_prefix=f"profile/validation{suffix}",
            )

    @torch.no_grad()
    def validate_and_log(
        self,
        model: ModelAdapter,
        step: int,
        scorer: Scorers | None,
        *,
        variant: str,
        metric_prefix: str = "val",
        image_name: str = "validation",
        profile_prefix: str = "profile/validation",
    ) -> None:
        """Run validation: sample images and optionally compute reward scores.

        Args:
            model: The model adapter (with transformer) to sample from.
            step: Current training step, used for logging.
            scorer: Scores each sample (with ``validation_log_rewards``) and
                logs the reward statistics.
            variant: The model variant the samples run under.

        Padding rows are sampled like every other row (the ranks' collectives
        stay balanced) but are neither logged nor scored.
        """
        if self._validation_dataloader is None or self._validation_stream is None:
            return

        logger.info(f"Validating at step {step}...")
        model.transformer.eval()
        if not self.validation_log_rewards:
            scorer = None

        base_seed = self.seed if self.validation_same_seed else self.seed + step
        epoch = 0 if self.validation_same_seed else step
        online = isinstance(self._validation_store, OnlineStore)

        def prepare(item: Row) -> Row:
            return deep_cast_float_dtype(
                self.prepare_row(item, mode="inference", epoch=epoch, online=online),
                model.dtype,
            )

        items = enumerate(
            item for batch in self.validation_dataloader for item in batch
        )
        started = time.perf_counter()
        groups = self.runner.submit(
            items,
            sample_one(self.validation_sampler, base_seed),
            score_one(scorer),
            prepare=prepare,
            k=1,
            epoch=epoch,
            variant=variant,
        ).wait()
        elapsed = time.perf_counter() - started

        image_count = 0
        scored: list[Row] = []
        for (row,) in groups:
            padding = is_padding(row)
            if (
                self.validation_log_images is True
                or image_count < self.validation_log_images
            ):
                prompt = row.get("prompt")
                image = (
                    None
                    if padding
                    else self.processor.annotate_output(cast(DecodedRow, row), row)
                    if self.validation_annotate_images
                    else row["clean_image"]
                )
                # Collective on every rank; None skips the emit.
                self.log_image(
                    image,
                    row[KEY],
                    step=step,
                    name=image_name,
                    caption=prompt if isinstance(prompt, str) else None,
                )
                image_count += self.world_size
            if not padding:
                scored.append(row)
        if scorer is not None:
            self.log_metrics(
                self._reward_metrics(scored, scorer, metric_prefix), step=step
            )
        self.log_phase_timing(
            "Validating", elapsed, len(groups), step, prefix=profile_prefix
        )

        model.transformer.train()
        logger.info(f"Completed validation at step {step}.")
