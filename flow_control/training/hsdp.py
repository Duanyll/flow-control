import math
import os
import shutil
from copy import deepcopy
from typing import Any

import aim
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate.utils import set_seed
from einops import reduce
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
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.datasets import DatasetConfig, collate_fn, parse_dataset
from flow_control.processors import Processor
from flow_control.samplers import Sampler
from flow_control.utils.common import (
    deep_move_to_device,
    tensor_to_pil,
)
from flow_control.utils.data import DistributedBucketSampler
from flow_control.utils.ema import apply_ema_maybe
from flow_control.utils.logging import console, get_logger, warn_once
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)
from flow_control.utils.weighting import LossWeighting, TimestepWeighting

logger = get_logger(__name__)


class HsdpTrainerConfig(BaseModel):
    model: ModelAdapter
    sampler: Sampler
    processor: Processor
    dataset: DatasetConfig
    sample_dataset: DatasetConfig | None = None
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    timestep_weighting: TimestepWeighting
    loss_weighting: LossWeighting

    checkpoint_dir: str
    seed_checkpoint_dir: str | None = None
    logging_dir: str = "."
    experiment_name: str
    resume_from_step: int | None = None
    checkpoint_steps: int = 500
    checkpoint_limit: int = 5
    sample_steps: int = 1000

    gradient_checkpointing: bool = True
    num_dataloader_workers: int = 4

    seed: int = 42
    global_batch_size: int = 16
    train_steps: int = 10000
    ema_decay: float = 0.999
    clip_grad_norm: float = 1.0

    hsdp_shard_dim: int = 1


class HsdpTrainer(Stateful):
    raw_conf: dict[str, Any]
    conf: HsdpTrainerConfig

    @property
    def model(self):
        return self.conf.model

    @property
    def sampler(self):
        return self.conf.sampler

    @property
    def processor(self):
        return self.conf.processor

    @property
    def transformer(self):
        return self.conf.model.transformer

    dataloader: StatefulDataLoader
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    sample_dataloader: StatefulDataLoader | None = None
    tracker: aim.Run

    current_step: int = 0

    world_size: int
    rank: int
    local_rank: int
    mesh: dist.device_mesh.DeviceMesh | None = None

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def is_local_main_process(self):
        return self.local_rank == 0

    @property
    def device(self):
        return torch.device(f"cuda:{self.local_rank}")

    @property
    def grad_acc_steps(self):
        return self.conf.global_batch_size // self.world_size

    @property
    def total_epochs(self):
        return math.ceil(self.conf.train_steps / len(self.dataloader))

    @property
    def current_epoch(self):
        return self.current_step // len(self.dataloader)

    def __init__(self, **kwargs):
        self.raw_conf = deepcopy(kwargs)
        self.conf = HsdpTrainerConfig(**kwargs)

    def init_device_mesh(self):
        if not dist.is_torchelastic_launched():
            raise RuntimeError("HSDPTrainer requires torchelastic launch.")
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(
                self.world_size // self.conf.hsdp_shard_dim,
                self.conf.hsdp_shard_dim,
            ),
            mesh_dim_names=("replicate", "shard"),
        )
        torch.cuda.set_device(self.local_rank)
        logger.info(
            f"Initialized device mesh: "
            f"world_size={self.world_size}, "
            f"replicate_dim={self.world_size // self.conf.hsdp_shard_dim}, "
            f"shard_dim={self.conf.hsdp_shard_dim}"
        )

    def init_tracker(self):
        if not self.is_main_process:
            return
        self.tracker = aim.Run(
            repo=self.conf.logging_dir, experiment=self.conf.experiment_name
        )
        self.tracker["hparams"] = self.raw_conf
        logger.info(
            f"Initialized Aim tracker at {self.conf.logging_dir}, experiment={self.conf.experiment_name}."
        )

    def load_transformer(self):
        with torch.device("meta"):
            self.model.load_transformer(use_meta_device=True)
            if self.conf.gradient_checkpointing:
                if (
                    hasattr(self.transformer, "_supports_gradient_checkpointing")
                    and self.transformer._supports_gradient_checkpointing
                ):
                    self.transformer.enable_gradient_checkpointing()
                else:
                    warn_once(
                        logger,
                        "Gradient checkpointing is enabled in the config, "
                        "but the transformer model does not support it.",
                    )
        if not hasattr(self.transformer, "_no_split_modules"):
            raise ValueError(
                "The transformer model must define _no_split_modules for HSDP."
            )
        fsdp_layers: list[str] = self.transformer._no_split_modules
        count = 0
        for _, module in self.transformer.named_modules():
            module_type = type(module).__name__
            if module_type in fsdp_layers:
                fully_shard(module, mesh=self.mesh)
                count += 1
        fully_shard(self.transformer, mesh=self.mesh)
        logger.info(f"Transformer is sharded with {count} FSDP layers.")

        if self.conf.seed_checkpoint_dir is not None:
            model_sd, _ = get_state_dict(
                self.transformer, [], options=StateDictOptions(strict=False)
            )
            dcp.load(
                model_sd,
                checkpoint_id=self.conf.seed_checkpoint_dir,
                planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
            )
            logger.info(
                f"Loaded seed checkpoint from {self.conf.seed_checkpoint_dir} into transformer."
            )
        elif not self.model.all_trainable:
            warn_once(
                logger,
                "Model is not fully trainable and no seed checkpoint is provided. "
                "This may lead to uninitialized parameters.",
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
        dataset: Any = parse_dataset(self.conf.dataset)
        if hasattr(dataset, "lengths"):
            dataset_lengths = dataset.lengths
            logger.info(
                f"Training Dataset has {len(dataset)} samples, separated into {len(dataset_lengths)} buckets."
            )
        else:
            dataset_lengths = None
            logger.info(f"Training Dataset has {len(dataset)} samples.")
        sampler = DistributedBucketSampler(
            lengths=dataset_lengths,
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
        self.processor.device = self.device
        self.processor.load_models(["decode"])
        dataset: Any = parse_dataset(self.conf.sample_dataset)
        dataset_length = len(dataset)
        logger.info(f"Sample Dataset has {dataset_length} samples.")
        sampler = DistributedBucketSampler(
            lengths=[dataset_length],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.conf.seed,
            grad_acc_steps=self.grad_acc_steps,
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

    def load_dcp_checkpoint(self, checkpoint_dir: str):
        state_dict = {"app": self}
        dcp.load(
            state_dict,
            checkpoint_id=checkpoint_dir,
            planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
        )
        logger.info(f"Resumed DCP checkpoint from {checkpoint_dir}.")

    def get_checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.conf.checkpoint_dir, f"step_{step:07d}")

    def log_images(self, image, key):
        # image may have different shapes, use gather_object to collect all images to main process
        image = image.cpu()
        if self.is_main_process:
            images = [None] * self.world_size
            dist.gather_object(image, images, dst=0)
            keys = [None] * self.world_size
            dist.gather_object(key, keys, dst=0)

            unique_images = {}
            for k, img in zip(keys, images, strict=True):
                unique_images[k] = img
            for k, img in unique_images.items():
                pil_image = tensor_to_pil(img)
                self.tracker.track(
                    aim.Image(pil_image), name=f"samples/{k}", step=self.current_step
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
        progress.start()
        self.transformer.eval()
        with apply_ema_maybe(self.optimizer):
            for batch in self.sample_dataloader:
                batch = deep_move_to_device(batch, self.device)
                generator = torch.Generator(device=self.device).manual_seed(
                    self.conf.seed
                )
                self.processor.initialize_latents(batch, generator=generator)
                clean_latents = self.sampler.sample(self.model, batch)
                image = self.processor.decode_output(clean_latents, batch)
                key = batch.get("__key__", "unknown")
                self.log_images(image, key)
                progress.advance(task)
        self.transformer.train()
        progress.stop()
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
        timesteps = self.conf.timestep_weighting.sample_timesteps(1)
        timesteps = timesteps.to(device=self.model.device, dtype=self.model.dtype)
        clean = batch["clean_latents"]
        noise = torch.randn_like(clean)
        batch["noisy_latents"] = (1.0 - timesteps) * clean + timesteps * noise

        model_pred = self.model.predict_velocity(batch, timesteps).float()
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
        dist.all_gather_object(loss, losses)
        avg_loss = sum(losses) / len(losses)  # type: ignore
        self.tracker.track(avg_loss, name="train/loss", step=self.current_step)
        self.tracker.track(lr, name="train/lr", step=self.current_step)
        return avg_loss, lr

    def rotate_checkpoints_maybe(self):
        if not self.is_main_process:
            return

        if self.conf.checkpoint_limit <= 0:
            return
        checkpoint_dirs = []
        for name in os.listdir(self.conf.checkpoint_dir):
            if name.startswith("step_"):
                checkpoint_dirs.append(name)
        if len(checkpoint_dirs) <= self.conf.checkpoint_limit:
            return
        checkpoint_dirs.sort()
        num_to_remove = len(checkpoint_dirs) - self.conf.checkpoint_limit
        for i in range(num_to_remove):
            dir_to_remove = os.path.join(self.conf.checkpoint_dir, checkpoint_dirs[i])
            shutil.rmtree(dir_to_remove)
            logger.info(f"Removed old checkpoint: {dir_to_remove}")

    def save_dcp_checkpoint(self):
        checkpoint_dir = self.get_checkpoint_dir(self.current_step)
        state_dict = {"app": self}
        dcp.save(
            state_dict,
            checkpoint_id=checkpoint_dir,
            planner=dcp.default_planner.DefaultSavePlanner(),
        )
        logger.info(
            f"Saved DCP checkpoint at step {self.current_step} to {checkpoint_dir}."
        )
        self.rotate_checkpoints_maybe()

    def cleanup(self):
        dist.barrier()
        dist.destroy_process_group()
        logger.info("Cleaned up distributed resources.")

    def train(self):
        set_seed(self.conf.seed)
        self.init_device_mesh()
        self.init_tracker()
        self.load_transformer()
        self.make_optimizer_and_scheduler()
        self.make_train_dataloader()
        self.make_sample_dataloader_maybe()

        if self.conf.resume_from_step is not None:
            self.load_dcp_checkpoint(
                self.get_checkpoint_dir(self.conf.resume_from_step)
            )

        self.sample_and_log()

        progress, task = self.make_train_progress_bar()
        progress.start()

        starting_epoch = self.current_epoch
        for _ in range(starting_epoch, self.total_epochs):
            total_loss = 0.0
            progress.update(task, epoch=self.current_epoch + 1)
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

                total_loss, lr = self.log_loss_lr(total_loss)
                self.current_step += 1
                progress.update(
                    task, completed=self.current_step, loss=total_loss, lr=lr
                )

                if (self.current_step % self.conf.checkpoint_steps == 0) or (
                    self.current_step == self.conf.train_steps
                ):
                    self.save_dcp_checkpoint()

                if self.current_step % self.conf.sample_steps == 0:
                    self.sample_and_log()

                if self.current_step >= self.conf.train_steps:
                    break

        progress.stop()
        logger.info("Training completed.")

        self.cleanup()
