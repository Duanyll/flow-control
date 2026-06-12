from collections.abc import Generator
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
from flow_control.datasets import DatasetConfig
from flow_control.processors import Processor
from flow_control.rewards import Reward, execute_reward
from flow_control.rewards.base import BaseReward
from flow_control.samplers import Sampler
from flow_control.samplers.sampler import derive_seed
from flow_control.utils.logging import console, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
)

from ..data import (
    DistributedBucketSampler,
    PaddingAwareDatasetWrapper,
    collate_fn,
    seed_worker,
)
from .hsdp import HsdpMixin
from .logging import LoggingMixin
from .preprocess import PreprocessMixin

logger = get_logger(__name__)


class ValidationMixin(PreprocessMixin, LoggingMixin, HsdpMixin, BaseModel):
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

    processor: Processor  # Shared property
    validation_sampler: Sampler

    # -------------------------------- Properties -------------------------------- #

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
        """Create the validation dataloader, loading decode models on the processor."""
        if self.validation_dataset is None:
            logger.info("No validation dataset configured, skipping.")
            return

        dataset = PaddingAwareDatasetWrapper(
            self.parse_inference_dataset(self.validation_dataset)
        )
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.seed,
            grad_acc_steps=1,
        )
        self._validation_dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.validation_num_workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )
        logger.info(f"Validation dataloader created with {len(dataset)} samples.")

    @torch.no_grad()
    def validate_and_log(
        self,
        model: ModelAdapter,
        step: int,
        reward: BaseReward | None = None,
    ) -> None:
        """Run validation: sample images and optionally compute reward scores.

        Args:
            model: The model adapter (with transformer) to sample from.
            sampler: The sampler to use for generating images.
            processor: The processor for latent init, negative batch, and decoding.
            step: Current training step, used for logging.
            reward: If provided, score each sample and log mean reward.
        """
        if self._validation_dataloader is None:
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
            disable=len(self.validation_dataloader) <= 1,
        )
        task = progress.add_task(
            "Validating",
            total=len(self.validation_dataloader),
        )

        def sample_submitter() -> Generator[tuple[dict[str, Any], str]]:
            image_count = 0
            with progress:
                for batch in self.validation_dataloader:
                    batch = deep_move_to_device(batch, self.device)
                    batch: Any = self.preprocess_for_inference(batch)
                    batch = deep_cast_float_dtype(batch, model.dtype)
                    negative_batch: Any = (
                        self.processor.get_negative_batch(batch)
                        if self.validation_sampler.cfg_scale > 1.0
                        else None
                    )
                    base_seed = (
                        self.seed if self.validation_same_seed else self.seed + step
                    )
                    key = batch.get("__key__", "unknown")
                    seed = derive_seed(base_seed, key)
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    self.processor.initialize_latents(
                        batch,
                        generator=generator,
                        device=self.device,
                        dtype=model.dtype,
                    )
                    sample_output = self.validation_sampler.sample(
                        model, batch, negative_batch=negative_batch, generator=generator
                    )
                    decoded = self.processor.decode_output(
                        sample_output.final_latents, batch
                    )
                    batch.update(decoded)

                    # Log image
                    if (
                        self.validation_log_images is True
                        or image_count < self.validation_log_images
                    ):
                        prompt = batch.get("prompt")
                        image = (
                            self.processor.annotate_output(decoded, batch)
                            if self.validation_annotate_images
                            else batch["clean_image"]
                        )
                        self.log_image(
                            image,
                            key,
                            step=step,
                            caption=prompt if isinstance(prompt, str) else None,
                        )
                        image_count += self.world_size

                    progress.advance(task)

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

            if reward_values:
                normalized = torch.cat([r.normalized for r in reward_values], dim=0)
                raw = torch.cat([r.raw for r in reward_values], dim=0)
                weights = reward_values[0].weights.to(
                    device=normalized.device,
                    dtype=normalized.dtype,
                )
                labels = reward_values[0].labels
                aggregate = (normalized * weights).sum(dim=-1)
                metrics: dict[str, float] = {
                    "val/reward_mean": aggregate.mean().item(),
                    "val/reward_std": aggregate.std(correction=0).item(),
                }
                for i, label in enumerate(labels):
                    metrics[f"val/raw/{label}_mean"] = raw[:, i].mean().item()
                    metrics[f"val/raw/{label}_std"] = raw[:, i].std(correction=0).item()
                    metrics[f"val/normalized/{label}_mean"] = (
                        normalized[:, i].mean().item()
                    )
                    metrics[f"val/normalized/{label}_std"] = (
                        normalized[:, i].std(correction=0).item()
                    )
                self.log_metrics(metrics, step=step)
        else:
            # Just iterate to generate and log images, no reward scoring
            for _ in sample_submitter():
                pass

        self.log_progress_timing(progress, step, prefix="profile/validation")

        model.transformer.train()
        logger.info(f"Completed validation at step {step}.")
