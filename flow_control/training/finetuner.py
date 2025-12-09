import copy
import os
import shutil
from typing import Any, cast

import aim
import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxControlPipeline
from diffusers.utils.torch_utils import is_compiled_module
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from flow_control.adapters import ModelAdapter
from flow_control.datasets import DatasetConfig, collate_fn, parse_dataset
from flow_control.processors import Processor
from flow_control.samplers import Sampler
from flow_control.utils.common import (
    deep_move_to_device,
    parse_checkpoint_step,
    tensor_to_pil,
)
from flow_control.utils.ema import EMA
from flow_control.utils.logging import console, get_logger
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)
from flow_control.utils.weighting import LossWeighting, TimestepWeighting

logger = get_logger(__name__)


class Fintuner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Utility classes
    model: ModelAdapter
    sampler: Sampler
    processor: Processor
    dataset: DatasetConfig
    sample_dataset: DatasetConfig | None = None
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    timestep_weighting: TimestepWeighting
    loss_weighting: LossWeighting

    _accelerator: Accelerator = PrivateAttr()
    _ema: EMA | None = PrivateAttr(default=None)

    # Loading, saving, and logging parameters
    output_dir: str
    logging_dir: str = "./runs"
    experiment_name: str
    resume_from_checkpoint: str | None = None
    checkpoint_steps: int = 500
    """ Save a checkpoint every N optimizer steps """
    sample_steps: int = 50
    """ Sample and log every N optimizer steps """
    checkpoint_limit: int | None = None
    """ Maximum number of checkpoints to keep. None means no limit """
    _resume_checkpoint_path: str | None = None
    _resume_checkpoint_step: int = 0

    @model_validator(mode="after")
    def _check_resume_from_checkpoint(self):
        if self.resume_from_checkpoint is None:
            return self

        if self.resume_from_checkpoint != "latest":
            path = os.path.basename(self.resume_from_checkpoint)
        else:
            dirs = os.listdir(self.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=parse_checkpoint_step)
            path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                raise ValueError(f"Checkpoint {path} not found")

        self._resume_checkpoint_step = parse_checkpoint_step(path)
        if self._resume_checkpoint_step < 0:
            raise ValueError(f"Cannot parse checkpoint step from {path}")
        self._resume_checkpoint_path = os.path.join(self.output_dir, path)

        return self

    # Other training parameters
    gradient_checkpointing: bool = True
    num_dataloader_workers: int = 4

    # Hyperparameters
    seed: int = 42
    global_batch_size: int = 1
    """ How many samples should the model see in one optimizer update across all devices """
    ema_decay: float = 0.999
    """ Set to 1.0 to disable EMA. Does not affect training but only used for sampling. """
    clip_grad_norm: float = 1.0

    # Property shortcuts
    @property
    def ema(self) -> EMA | None:
        return self._ema

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator

    @property
    def transformer(self) -> Any:
        return self.model.transformer

    @property
    def device(self):
        return self.accelerator.device

    @property
    def train_steps(self) -> int:
        return self.scheduler["num_training_steps"]

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.global_batch_size // self.accelerator.num_processes

    def __init__(self, **kwargs):
        conf = copy.deepcopy(kwargs)

        super().__init__(**kwargs)

        self._accelerator = Accelerator(
            mixed_precision="no",
            log_with="aim",
            project_config=ProjectConfiguration(
                project_dir=self.output_dir, logging_dir=self.logging_dir
            ),
        )
        if self.global_batch_size % self.accelerator.num_processes != 0:
            raise ValueError(
                f"Global batch size {self.global_batch_size} must be divisible"
                f" by number of processes {self.accelerator.num_processes}"
            )
        else:
            logger.info(
                f"Global batch size: {self.global_batch_size}, "
                f"gradient accumulation steps: {self.gradient_accumulation_steps}"
            )
        self.accelerator.gradient_accumulation_steps = self.gradient_accumulation_steps
        self.accelerator.init_trackers(
            self.experiment_name,
            config=conf,
        )
        logger.info(f"Experiment name: {self.experiment_name}")
        logger.info(f"Saving model to: {self.output_dir}")
        logger.info(f"Saving logs to: {self.logging_dir}")

    def _unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def _save_model_hook(self, models, weights, output_dir):
        if not self.accelerator.is_main_process:
            return

        layers_to_save = self.model.save_model()

        if weights:
            # Remove the model from input list to avoid default saving behavior
            weights.pop()

        # We save the lora with FluxControlPipeline so it can be directly loaded
        # by the pipeline for inference
        FluxControlPipeline.save_lora_weights(
            output_dir, transformer_lora_layers=layers_to_save
        )
        logger.info(f"Saved model to {output_dir}")

        if self.ema is not None:
            ema_layers = self.ema.state_dict()
            FluxControlPipeline.save_lora_weights(
                os.path.join(output_dir, "ema"),
                transformer_lora_layers=ema_layers,  # type: ignore
            )
            logger.info(f"Saved EMA model to {output_dir}")

    def _load_weights(self, input_dir):
        lora_state_dict = cast(dict, FluxControlPipeline.lora_state_dict(input_dir))
        self.model.load_model(lora_state_dict)
        logger.info(f"Loaded model from {input_dir}")

    def _load_model_hook(self, models, input_dir):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.model.load_transformer()

        self._load_weights(input_dir)

        if self.ema is not None:
            ema_state_dict = FluxControlPipeline.lora_state_dict(
                os.path.join(input_dir, "ema")
            )
            self.ema.load_state_dict(ema_state_dict)
            self.ema.to(self.device)
            logger.info(f"Loaded EMA model from {input_dir}")

    def _load_models_for_training(self):
        self.model.load_transformer()
        if self.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.processor.load_models(["decode"], device=self.device)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        optimizer = parse_optimizer(self.optimizer, params)
        num_trainable_params = sum(p.numel() for p in params)
        logger.info(
            f"{optimizer.__class__.__name__} created with "
            f"{num_trainable_params / (1024 * 1024):.2f}M trainable parameters"
        )
        return optimizer

    def _make_dataloader(self) -> torch.utils.data.DataLoader:
        with self.accelerator.main_process_first():
            dataset = parse_dataset(self.dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            collate_fn=collate_fn,
        )
        return dataloader

    def _make_sample_dataloader(self) -> torch.utils.data.DataLoader | None:
        if self.sample_dataset is None:
            return None

        with self.accelerator.main_process_first():
            dataset = parse_dataset(self.sample_dataset)
        sample_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            collate_fn=collate_fn,
        )
        return sample_dataloader

    def _make_ema(self):
        if self.ema_decay != 1.0:
            self._ema = EMA(self.transformer, self.ema_decay)
            self._ema.register()
            logger.info(f"EMA created with decay {self.ema_decay}")

    def _try_resume_from_checkpoint(self) -> int:
        if self._resume_checkpoint_path is None:
            return 0
        self.accelerator.load_state(self._resume_checkpoint_path)
        logger.info(f"Resumed from checkpoint {self._resume_checkpoint_path}")
        return self._resume_checkpoint_step

    def _try_remove_extra_checkpoints(self):
        if self.checkpoint_limit is None:
            return

        checkpoints = [
            d for d in os.listdir(self.output_dir) if d.startswith("checkpoint")
        ]

        if len(checkpoints) <= self.checkpoint_limit:
            return

        checkpoints = sorted(
            checkpoints,
            key=lambda x: parse_checkpoint_step(x),
            reverse=True,
        )

        for checkpoint in checkpoints[self.checkpoint_limit :]:
            checkpoint_path = os.path.join(self.output_dir, checkpoint)
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed checkpoint {checkpoint_path}")

    def _save_checkpoint(self, current_step):
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._try_remove_extra_checkpoints()

        if (
            self.accelerator.is_main_process
            or self.accelerator.distributed_type == DistributedType.DEEPSPEED
        ):
            save_path = os.path.join(self.output_dir, f"checkpoint-{current_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved checkpoint to {save_path} at step {current_step}")

    def _final_save(self):
        if self.accelerator.is_main_process:
            self._save_model_hook([self.transformer], [], self.output_dir)
            logger.info(f"Final model saved to {self.output_dir}")

    def _train_step(self, batch) -> torch.Tensor:
        timesteps = self.timestep_weighting.sample_timesteps(1)
        timesteps = timesteps.to(device=self.model.device, dtype=self.model.dtype)
        loss = self.model.train_step(batch, timesteps)
        weighting = self.loss_weighting.get_weights(timesteps)
        weighting = weighting.to(device=loss.device, dtype=loss.dtype)
        weighted_loss = (loss * weighting).mean()
        return weighted_loss

    def _check_loss_validity(self, loss, global_step, batch):
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            rank = self.accelerator.process_index
            logger.error(f"Step {global_step} on rank {rank} loss is NaN or Inf")
            save_path = os.path.join(self.output_dir, f"dump-{rank}-{global_step}")
            self.accelerator.save_state(save_path)
            torch.save(batch, os.path.join(save_path, "batch.pt"))
            logger.info(f"Dumped batch to {save_path}")
            self.accelerator.end_training()
            raise ValueError(f"Step {global_step} on rank {rank} loss is NaN or Inf")

    @torch.no_grad()
    def _sample_and_log(self, dataloader, current_step: int):
        self.transformer.eval()
        if dataloader is None:
            return

        if self.ema is not None:
            self.ema.apply_shadow()

        logger.info(f"Sampling at step {current_step}...")

        for batch in dataloader:
            batch = deep_move_to_device(batch, self.device)
            clean_latents = self.sampler.sample(self.model, batch)
            image = self.processor.decode_output(clean_latents, batch)  # type: ignore
            key = batch.get("__key__", "unknown")
            gathered_images = self.accelerator.gather_for_metrics(
                image, use_gather_object=True
            )
            gathered_keys = self.accelerator.gather_for_metrics(
                key, use_gather_object=True
            )
            if self.accelerator.is_main_process:
                # Deduplicate gathered results by keys
                unique_dict = {}
                for k, img in zip(gathered_keys, gathered_images, strict=False):
                    unique_dict[k] = img
                for k, img in unique_dict.items():
                    pil_image = tensor_to_pil(img)
                    # FIXME: Broken logging key in aim
                    self.accelerator.log(
                        {f"sample/{k}": aim.Image(pil_image)}, step=current_step
                    )

        if self.ema is not None:
            self.ema.restore()

        self.transformer.train()
        self.accelerator.wait_for_everyone()
        logger.info(f"Sampling at step {current_step} done.")

    def _make_progress_bar(self, starting_step: int, starting_epoch: int):
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" Epoch: {task.fields[epoch]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("• Loss: {task.fields[loss]:.4f}"),
            TextColumn("• LR: {task.fields[lr]:.6f}"),
            console=console,
        )
        task = progress.add_task(
            description="Training",
            total=self.train_steps,
            completed=starting_step,
            epoch=starting_epoch,
            loss=0.0,
            lr=0.0,
        )
        return progress, task

    def train(self):
        set_seed(self.seed)
        self._load_models_for_training()

        optimizer = self._make_optimizer()
        dataloader = self._make_dataloader()
        lr_scheduler = parse_scheduler(self.scheduler, optimizer)
        self.model.transformer, optimizer, dataloader, lr_scheduler = (
            self.accelerator.prepare(
                self.model.transformer, optimizer, dataloader, lr_scheduler
            )
        )

        sample_dataloader = self._make_sample_dataloader()
        if sample_dataloader is not None:
            self.accelerator.prepare(sample_dataloader)

        self._make_ema()
        starting_step = self._try_resume_from_checkpoint()
        starting_epoch = starting_step // len(dataloader)
        total_epochs = self.train_steps // len(dataloader) + 1
        current_step = starting_step

        self._sample_and_log(sample_dataloader, current_step)

        logger.info(
            f"Starting training from step {starting_step}, epoch {starting_epoch}/{total_epochs}"
        )
        progress, progress_task = self._make_progress_bar(starting_step, starting_epoch)
        progress.start()

        for epoch in range(starting_epoch, total_epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            for _step, batch in enumerate(dataloader):
                batch = deep_move_to_device(batch, self.device)

                with self.accelerator.accumulate(self.transformer):
                    loss = self._train_step(batch) / self.gradient_accumulation_steps
                    self._check_loss_validity(loss, current_step, batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(), self.clip_grad_norm
                        )

                    optimizer.step()
                    optimizer.zero_grad()

                    if self.accelerator.sync_gradients and self.ema is not None:
                        logs = {
                            "loss": loss.item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        self.accelerator.log(logs, step=current_step)
                        progress.update(progress_task, advance=1, epoch=epoch, **logs)
                        current_step += 1
                        self.ema.update()
                        lr_scheduler.step()

                if current_step % self.sample_steps == 0:
                    self._sample_and_log(sample_dataloader, current_step)

                if current_step % self.checkpoint_steps == 0:
                    self._save_checkpoint(current_step)

                if current_step >= self.train_steps:
                    break

        progress.stop()
        self._final_save()

        logger.info("Training completed.")
        self.accelerator.end_training()
