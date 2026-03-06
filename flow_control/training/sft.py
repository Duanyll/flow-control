import math
import time
from typing import Any

import torch
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
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.utils.common import (
    deep_cast_float_dtype,
    deep_move_to_device,
)
from flow_control.utils.ema import apply_ema_maybe
from flow_control.utils.logging import (
    console,
    dump_if_failed,
    get_logger,
    warn_once,
)

from .data import DistributedBucketSampler, PaddingAwareDatasetWrapper, collate_fn
from .hsdp_engine import (
    DistributedExitSignal,
    distributed_main,
)
from .trainer_base import HsdpTrainerBase, HsdpTrainerBaseConfig
from .weighting import (
    LogitNormalTimestepWeighting,
    LossWeighting,
    TimestepWeighting,
    UniformLossWeighting,
)

logger = get_logger(__name__)


class HsdpTrainerConfig(HsdpTrainerBaseConfig):
    timestep_weighting: TimestepWeighting = LogitNormalTimestepWeighting()
    loss_weighting: LossWeighting = UniformLossWeighting()

    cfg_drop_prob: float = 0.0
    latent_length_test_mode: bool = False


class HsdpSftTrainer(HsdpTrainerBase[HsdpTrainerConfig]):
    conf: HsdpTrainerConfig

    @property
    def grad_acc_steps(self):
        return self.conf.global_batch_size // self.world_size

    @property
    def total_epochs(self):
        return math.ceil(
            self.conf.train_steps / (len(self.dataloader) // self.grad_acc_steps)
        )

    @property
    def current_epoch(self):
        return self.current_step // (len(self.dataloader) // self.grad_acc_steps)

    def __init__(self, **kwargs):
        self.conf = HsdpTrainerConfig(**kwargs)
        super().__init__(**kwargs)

    def make_train_dataloader(self):
        dataset = PaddingAwareDatasetWrapper(self._parse_dataset(self.conf.dataset))
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

    @staticmethod
    def _parse_dataset(config):
        from flow_control.datasets import parse_dataset

        return parse_dataset(config)

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
        import torch.distributed as dist

        lr = self.scheduler.get_last_lr()[0]
        losses = [None] * self.world_size
        dist.all_gather_object(losses, loss)
        avg_loss = sum(losses) / len(losses)  # type: ignore
        if self.is_main_process:
            self.tracker.track(avg_loss, name="train/loss", step=self.current_step)
            self.tracker.track(lr, name="train/lr", step=self.current_step)
        return avg_loss, lr

    def _after_sync_step(self, total_loss: float, progress, task, exit_signal) -> bool:
        """Handle optimizer step, logging, checkpointing after a gradient sync.

        Returns True if training should stop early.
        """
        if self.conf.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), self.conf.clip_grad_norm
            )

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        avg_loss, lr = self.log_loss_lr(total_loss)
        self.current_step += 1
        progress.update(task, loss=avg_loss, lr=lr)
        progress.advance(task)

        if exit_signal:
            logger.info(
                "Exit signal received. Saving checkpoint and exiting training loop..."
            )
            self.save_dcp_checkpoint(self.get_checkpoint_dir(self.current_step))
            return True

        if (self.current_step % self.conf.checkpoint_steps == 0) or (
            self.current_step == self.conf.train_steps
        ):
            self.save_dcp_checkpoint(self.get_checkpoint_dir(self.current_step))
            self.rotate_checkpoints_maybe()

        if self.current_step % self.conf.sample_steps == 0:
            self.sample_and_log()

        return self.current_step >= self.conf.train_steps

    def check_loss(self, loss: torch.Tensor):
        if not torch.isfinite(loss):
            logger.error(
                f"Non-finite loss detected (loss={loss.item()}). Stopping training."
            )
            raise RuntimeError("Non-finite loss detected.")

    @distributed_main
    def run(self):
        if self.conf.latent_length_test_mode:
            self.run_latent_length_test()
            return

        self.set_seed()
        self.init_tracker()
        self.load_transformer(self.model, self.conf.seed_checkpoint_dir)
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
                    with dump_if_failed(logger, batch):
                        is_sync_step = (i + 1) % self.grad_acc_steps == 0
                        self.transformer.set_requires_gradient_sync(is_sync_step)

                        batch = deep_move_to_device(batch, self.device)
                        loss = self.train_step(batch) / self.grad_acc_steps
                        self.check_loss(loss)
                        loss.backward()

                        total_loss += loss.item()

                    if not is_sync_step:
                        continue

                    should_stop = self._after_sync_step(
                        total_loss, progress, task, exit_signal
                    )
                    total_loss = 0.0

                    if should_stop:
                        if exit_signal:
                            return
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
        self.load_transformer(self.model)
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
