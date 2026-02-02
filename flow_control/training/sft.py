import math
import os
import shutil
import time
from typing import Any

import aim
import numpy as np
import torch
import torch.distributed as dist
from einops import reduce
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.datasets import DatasetConfig, collate_fn, parse_dataset
from flow_control.processors import Processor
from flow_control.samplers import Sampler, SimpleEulerSampler
from flow_control.utils.common import (
    deep_cast_float_dtype,
    deep_move_to_device,
    tensor_to_pil,
)
from flow_control.utils.data import DistributedBucketSampler, PaddingAwareDatasetWrapper
from flow_control.utils.ema import apply_ema_maybe
from flow_control.utils.logging import console, get_logger, get_version, warn_once
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)
from flow_control.utils.weighting import (
    LogitNormalTimestepWeighting,
    LossWeighting,
    TimestepWeighting,
    UniformLossWeighting,
)

from .hsdp_engine import (
    DistributedExitSignal,
    HsdpEngine,
    HsdpEngineConfig,
    distributed_main,
)

logger = get_logger(__name__)


class HsdpTrainerConfig(HsdpEngineConfig):
    model: ModelAdapter
    sampler: Sampler = SimpleEulerSampler()
    processor: Processor
    dataset: DatasetConfig
    sample_dataset: DatasetConfig | None = None
    optimizer: OptimizerConfig = {"class_name": "AdamW", "lr": 1e-4}
    scheduler: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}
    timestep_weighting: TimestepWeighting = LogitNormalTimestepWeighting()
    loss_weighting: LossWeighting = UniformLossWeighting()

    checkpoint_root: str
    seed_checkpoint_dir: str | None = None
    logging_dir: str = "."
    experiment_name: str
    resume_from_dir: str | None = None
    checkpoint_steps: int = 500
    checkpoint_limit: int = 5
    sample_steps: int = 1000

    num_dataloader_workers: int = 4

    global_batch_size: int = 16
    train_steps: int = 10000
    ema_decay: float = 0.999
    clip_grad_norm: float = 1.0
    cfg_drop_prob: float = 0.0

    latent_length_test_mode: bool = False


class HsdpSftTrainer(HsdpEngine, Stateful):
    conf: HsdpTrainerConfig

    @property
    def model(self):
        return self.conf.model

    @property
    def transformer(self):
        return self.conf.model.transformer

    @property
    def sampler(self):
        return self.conf.sampler

    @property
    def processor(self):
        return self.conf.processor

    dataloader: StatefulDataLoader
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    sample_dataloader: StatefulDataLoader | None = None
    tracker: aim.Run

    current_step: int = 0

    @property
    def grad_acc_steps(self):
        return self.conf.global_batch_size // self.world_size

    @property
    def total_epochs(self):
        # For consistency, we always count optimizer steps towards epochs
        # Multiple grad acc steps are counted as one optimizer step
        return math.ceil(
            self.conf.train_steps / (len(self.dataloader) // self.grad_acc_steps)
        )

    @property
    def current_epoch(self):
        return self.current_step // (len(self.dataloader) // self.grad_acc_steps)

    def __init__(self, **kwargs):
        self.conf = HsdpTrainerConfig(**kwargs)  # type: ignore
        super().__init__(**kwargs)

    def init_tracker(self):
        if not self.is_main_process:
            return
        self.tracker = aim.Run(
            repo=self.conf.logging_dir, experiment=self.conf.experiment_name
        )
        conf_dump = self.conf.model_dump(mode="json", warnings="none")
        conf_dump["__version__"] = get_version()
        self.tracker["hparams"] = conf_dump
        logger.info(
            f"Initialized Aim tracker at {self.conf.logging_dir}, experiment={self.conf.experiment_name}."
        )

    def make_optimizer_and_scheduler(self):
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = parse_optimizer(
            self.conf.optimizer, params, ema_decay=self.conf.ema_decay
        )
        num_trainable_params = sum(p.numel() for p in params)
        logger.info(
            f"Created optimizer with {num_trainable_params / 1e6:.2f}M trainable parameters."
        )
        self.scheduler = parse_scheduler(self.conf.scheduler, self.optimizer)

    def make_train_dataloader(self):
        dataset: Any = PaddingAwareDatasetWrapper(parse_dataset(self.conf.dataset))
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.conf.seed,
            grad_acc_steps=self.grad_acc_steps,
        )
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.conf.num_dataloader_workers,
            collate_fn=collate_fn,
        )

    def make_sample_dataloader_maybe(self):
        if self.conf.sample_dataset is None:
            logger.info("No sample dataset provided, skipping sample dataloader.")
            self.sample_dataloader = None
            return
        self.processor.load_models("decode", device=self.device)
        dataset: Any = PaddingAwareDatasetWrapper(
            parse_dataset(self.conf.sample_dataset)
        )
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.conf.seed,
            grad_acc_steps=1,
        )
        self.sample_dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.conf.num_dataloader_workers,
            collate_fn=collate_fn,
        )

    def state_dict(self):
        transformer_state_dict, optimizer_state_dict = get_state_dict(
            self.transformer, [self.optimizer], options=StateDictOptions(strict=False)
        )
        transformer_state_dict = self.model.filter_state_dict(transformer_state_dict)
        return {
            "transformer": transformer_state_dict,
            "optimizer": optimizer_state_dict,
            "dataloader": self.dataloader.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.transformer,
            [self.optimizer],
            model_state_dict=state_dict["transformer"],
            optim_state_dict=state_dict["optimizer"],
            options=StateDictOptions(strict=False),
        )
        self.dataloader.load_state_dict(state_dict["dataloader"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.current_step = state_dict["current_step"]

    def get_checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.conf.checkpoint_root, f"step_{step:07d}")

    def log_images(self, image, key):
        image = np.array(image)
        if self.is_main_process:
            images: list[Any] = [None] * self.world_size
            dist.gather_object(image, images, dst=0)
            keys = [None] * self.world_size
            dist.gather_object(key, keys, dst=0)

            for k, img in zip(keys, images, strict=True):
                if k == "__padding__":
                    continue
                self.tracker.track(
                    aim.Image(img), name=f"samples/{k}", step=self.current_step
                )
        else:
            dist.gather_object(image, None, dst=0)
            dist.gather_object(key, None, dst=0)
            return

    def make_sample_progress_bar(self):
        if self.sample_dataloader is None:
            raise RuntimeError("Sample dataloader is not initialized.")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" Batch: {task.completed}/{task.total}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
            disable=len(self.sample_dataloader) <= 1,
        )
        task = progress.add_task(
            "Sampling...",
            total=len(self.sample_dataloader),
        )
        return progress, task

    @torch.no_grad()
    def sample_and_log(self):
        if self.sample_dataloader is None:
            return

        logger.info(f"Sampling at step {self.current_step}...")
        progress, task = self.make_sample_progress_bar()
        self.transformer.eval()
        with apply_ema_maybe(self.optimizer), progress:
            for batch in self.sample_dataloader:
                batch = deep_cast_float_dtype(batch, self.model.dtype)
                batch = deep_move_to_device(batch, self.device)
                negative_batch: Any = (
                    self.processor.get_negative_batch(batch)
                    if self.sampler.cfg_scale > 1.0
                    else None
                )
                generator = torch.Generator(device=self.device).manual_seed(
                    self.conf.seed
                )
                self.processor.initialize_latents(
                    batch,
                    generator=generator,
                    device=self.device,
                    dtype=self.model.dtype,
                )
                clean_latents = self.sampler.sample(
                    self.model, batch, negative_batch=negative_batch
                )
                image = tensor_to_pil(
                    self.processor.decode_output(clean_latents, batch)
                )
                key = batch.get("__key__", "unknown")
                self.log_images(image, key)
                progress.advance(task)
        self.transformer.train()
        logger.info(f"Completed sampling at step {self.current_step}.")

    def make_train_progress_bar(self):
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" Epoch: {task.fields[epoch]}/{task.fields[total_epochs]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("• Loss: {task.fields[loss]:.4f}"),
            TextColumn("• LR: {task.fields[lr]:.6f}"),
            console=console,
        )
        task = progress.add_task(
            "Training...",
            total=self.conf.train_steps,
            completed=self.current_step,
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            loss=0.0,
            lr=0.0,
        )
        return progress, task

    def train_step(self, batch: Any):
        if (
            self.conf.cfg_drop_prob > 0.0
            and torch.rand(1).item() < self.conf.cfg_drop_prob
        ):
            negative_batch = self.processor.get_negative_batch(batch)
            if negative_batch is not None:
                batch = negative_batch
            else:
                warn_once(
                    logger,
                    f"CFG drop prob is set to {self.conf.cfg_drop_prob}, but no negative (unconditional) batch available.",
                )

        timesteps = self.conf.timestep_weighting.sample_timesteps(1)
        timesteps = timesteps.to(device=self.device, dtype=torch.float32)
        clean = batch["clean_latents"].float()
        noise = torch.randn_like(clean, dtype=torch.float32)
        batch["noisy_latents"] = (1.0 - timesteps) * clean + timesteps * noise
        batch = deep_cast_float_dtype(batch, self.model.dtype)

        model_pred = self.model.predict_velocity(
            batch, timesteps.to(dtype=self.model.dtype)
        ).float()
        target = noise.float() - clean.float()
        loss = (model_pred - target) ** 2
        loss = reduce(loss, "b n d -> 1", reduction="mean")

        weighting = self.conf.loss_weighting.get_weights(timesteps)
        weighting = weighting.to(device=loss.device, dtype=loss.dtype)
        weighted_loss = (loss * weighting).mean()
        return weighted_loss

    def log_loss_lr(self, loss: float):
        lr = self.scheduler.get_last_lr()[0]
        losses = [None] * self.world_size
        dist.all_gather_object(losses, loss)
        avg_loss = sum(losses) / len(losses)  # type: ignore
        if self.is_main_process:
            self.tracker.track(avg_loss, name="train/loss", step=self.current_step)
            self.tracker.track(lr, name="train/lr", step=self.current_step)
        return avg_loss, lr

    def rotate_checkpoints_maybe(self):
        if not self.is_main_process:
            return

        if self.conf.checkpoint_limit <= 0:
            return
        checkpoint_dirs = []
        for name in os.listdir(self.conf.checkpoint_root):
            if name.startswith("step_"):
                checkpoint_dirs.append(name)
        if len(checkpoint_dirs) <= self.conf.checkpoint_limit:
            return
        checkpoint_dirs.sort()
        num_to_remove = len(checkpoint_dirs) - self.conf.checkpoint_limit
        for i in range(num_to_remove):
            dir_to_remove = os.path.join(self.conf.checkpoint_root, checkpoint_dirs[i])
            shutil.rmtree(dir_to_remove)
            logger.info(f"Removed old checkpoint: {dir_to_remove}")

    @distributed_main
    def run(self):
        if self.conf.latent_length_test_mode:
            self.run_latent_length_test()
            return

        self.set_seed()
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.conf.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.make_train_dataloader()
        self.make_sample_dataloader_maybe()

        if self.conf.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.conf.resume_from_dir)

        self.sample_and_log()

        console.rule("[bold blue]Starting training[/bold blue]")

        progress, task = self.make_train_progress_bar()

        with progress, DistributedExitSignal(self) as exit_signal:
            starting_epoch = self.current_epoch
            for _ in range(starting_epoch, self.total_epochs):
                total_loss = 0.0
                progress.update(task, epoch=self.current_epoch + 1)
                if hasattr(self.dataloader.sampler, "set_epoch"):
                    self.dataloader.sampler.set_epoch(self.current_epoch)  # type: ignore
                for i, batch in enumerate(self.dataloader):
                    is_sync_step = (i + 1) % self.grad_acc_steps == 0
                    self.transformer.set_requires_gradient_sync(is_sync_step)

                    batch = deep_move_to_device(batch, self.device)
                    loss = self.train_step(batch) / self.grad_acc_steps
                    loss.backward()

                    total_loss += loss.item()

                    if not is_sync_step:
                        continue

                    if self.conf.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.transformer.parameters(), self.conf.clip_grad_norm
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    avg_loss, lr = self.log_loss_lr(total_loss)
                    total_loss = 0.0
                    self.current_step += 1
                    progress.update(task, loss=avg_loss, lr=lr)
                    progress.advance(task)

                    if exit_signal:
                        logger.info(
                            "Exit signal received. Saving checkpoint and exiting training loop..."
                        )
                        self.save_dcp_checkpoint(
                            self.get_checkpoint_dir(self.current_step)
                        )
                        return

                    if (self.current_step % self.conf.checkpoint_steps == 0) or (
                        self.current_step == self.conf.train_steps
                    ):
                        self.save_dcp_checkpoint(
                            self.get_checkpoint_dir(self.current_step)
                        )
                        self.rotate_checkpoints_maybe()

                    if self.current_step % self.conf.sample_steps == 0:
                        self.sample_and_log()

                    if self.current_step >= self.conf.train_steps:
                        break

        with apply_ema_maybe(self.optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self.current_step) + "_final"
            )

        console.rule("[bold green]Training completed[/bold green]")

    def run_latent_length_test(self):
        logger.warning(
            "Running in latent length test mode since enabled in config. This will not perform training, but will test "
            "increasing latent lengths until OOM. This is useful for finding the maximum latent length that fits in memory."
        )

        self.set_seed()
        self.load_transformer_from_seed(self.model)
        self.make_optimizer_and_scheduler()

        console.rule("[bold blue]Starting latent length test[/bold blue]")

        current_len = 0
        best_len = 0
        try:
            for batch in self.model.latent_length_test():
                current_len = batch["latent_length"]
                start_time = time.time()
                logger.info(f"Testing latent length: {current_len}")
                batch = deep_cast_float_dtype(batch, self.model.dtype)
                batch = deep_move_to_device(batch, self.device)
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Successfully trained with latent length {current_len} in {elapsed_time:.2f} seconds."
                )
                best_len = current_len
            logger.info(
                f"Latent length test completed successfully up to length {current_len}."
            )
        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"Out of memory error encountered at latent length {current_len}."
            )
        finally:
            console.rule("[bold red]Latent length test completed[/bold red]")
            console.print(Panel.fit(f"Maximum latent length: [bold]{best_len}[/bold]"))
