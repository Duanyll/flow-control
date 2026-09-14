from collections.abc import Generator, Iterator
from typing import Any, Literal

import torch
from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
from flow_control.rewards import Reward, execute_reward
from flow_control.rewards.base import BaseReward, RewardResult
from flow_control.samplers import Sampler, SampleRequest, derive_seed
from flow_control.utils.logging import console, get_logger
from flow_control.utils.tensor import deep_cast_float_dtype

from .base import BaseTrainer
from .data import DataMixin
from .logging import LoggingMixin

logger = get_logger(__name__)


class ValidationMixin(DataMixin, LoggingMixin, BaseTrainer, BaseModel):
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
    validation_reward: Reward | Literal[False] | None = None
    seed: int = 42

    validation_sampler: Sampler

    # -------------------------------- Properties -------------------------------- #

    _validation_store: RowStore | None = None
    _validation_stream: RowStream | None = None
    _validation_dataloader: StatefulDataLoader | None = None

    @property
    def validation_dataloader(self) -> StatefulDataLoader:
        if self._validation_dataloader is None:
            raise ValueError("Validation dataloader not created yet.")
        return self._validation_dataloader

    # ---------------------------------- Methods --------------------------------- #

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

    def _validation_requests(
        self, model: ModelAdapter, step: int
    ) -> Iterator[SampleRequest]:
        base_seed = self.seed if self.validation_same_seed else self.seed + step
        epoch = 0 if self.validation_same_seed else step
        online = isinstance(self._validation_store, OnlineStore)
        for items in self.validation_dataloader:
            for item in items:
                batch = self.prepare_row(
                    item, mode="inference", epoch=epoch, online=online
                )
                batch = deep_cast_float_dtype(batch, model.dtype)
                generator = torch.Generator(device=self.device).manual_seed(
                    derive_seed(base_seed, batch[KEY])
                )
                self.processor.initialize_latents(
                    batch,
                    generator=generator,
                    device=self.device,
                    dtype=model.dtype,
                )
                yield self.build_sample_request(
                    self.validation_sampler, batch, generator
                )

    @staticmethod
    def _reward_metrics(
        reward_values: list[RewardResult], metric_prefix: str
    ) -> dict[str, float]:
        """Per-rank reward statistics; empty when this rank scored nothing (all
        its rows were padding), so ``log_metrics`` still runs as a collective."""
        if not reward_values:
            return {}
        normalized = torch.cat([r.normalized for r in reward_values], dim=0)
        raw = torch.cat([r.raw for r in reward_values], dim=0)
        weights = reward_values[0].weights.to(
            device=normalized.device,
            dtype=normalized.dtype,
        )
        labels = reward_values[0].labels
        aggregate = (normalized * weights).sum(dim=-1)
        metrics: dict[str, float] = {
            f"{metric_prefix}/reward_mean": aggregate.mean().item(),
            f"{metric_prefix}/reward_std": aggregate.std(correction=0).item(),
        }
        for i, label in enumerate(labels):
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

    @torch.no_grad()
    def validate_and_log(
        self,
        model: ModelAdapter,
        step: int,
        reward: BaseReward | None = None,
        metric_prefix: str = "val",
        image_name: str = "validation",
        profile_prefix: str = "profile/validation",
    ) -> None:
        """Run validation: sample images and optionally compute reward scores.

        Args:
            model: The model adapter (with transformer) to sample from.
            step: Current training step, used for logging.
            reward: If provided, score each sample and log mean reward.

        Padding rows are sampled like every other row (the ranks' collectives
        stay balanced) but are neither logged nor scored.
        """
        if self._validation_dataloader is None or self._validation_stream is None:
            return

        logger.info(f"Validating at step {step}...")
        model.transformer.eval()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description:<20}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
            disable=len(self._validation_stream) <= 1,
        )
        task = progress.add_task("Validating", total=len(self._validation_stream))

        def sample_submitter() -> Generator[tuple[dict[str, Any], str]]:
            image_count = 0
            with progress:
                for run in self.validation_sampler.sample(
                    model, self._validation_requests(model, step)
                ):
                    batch: Any = run.batch
                    key = batch[KEY]
                    padding = is_padding(batch)
                    decoded = self.processor.decode_output(run.ctx.latents, batch)
                    batch.update(decoded)

                    if (
                        self.validation_log_images is True
                        or image_count < self.validation_log_images
                    ):
                        prompt = batch.get("prompt")
                        image = (
                            None
                            if padding
                            else self.processor.annotate_output(decoded, batch)
                            if self.validation_annotate_images
                            else batch["clean_image"]
                        )
                        # Collective on every rank; None skips the emit.
                        self.log_image(
                            image,
                            key,
                            step=step,
                            name=image_name,
                            caption=prompt if isinstance(prompt, str) else None,
                        )
                        image_count += self.world_size

                    progress.advance(task)
                    if not padding:
                        yield batch, key

        metric_reward = (
            None
            if self.validation_reward is False
            else self.validation_reward or reward
        )
        if metric_reward is not None and self.validation_log_rewards:
            # Use execute_reward to score all samples (with async overlap if supported)
            reward_values = execute_reward(
                metric_reward,
                sample_submitter(),
                lambda _tag, r: r,
            )
            self.log_metrics(
                self._reward_metrics(reward_values, metric_prefix), step=step
            )
        else:
            # Just iterate to generate and log images, no reward scoring
            for _ in sample_submitter():
                pass

        self.log_progress_timing(progress, step, prefix=profile_prefix)

        model.transformer.train()
        logger.info(f"Completed validation at step {step}.")
