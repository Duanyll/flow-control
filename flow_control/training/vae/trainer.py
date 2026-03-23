"""RGBA VAE trainer using AlphaVAE's multi-objective training algorithm.

Supports AutoencoderKL and AutoencoderKLFlux2 with FSDP2-based distributed
training, gradient accumulation, and optional PatchGAN discriminator.
"""

import math
import os
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from pydantic import ConfigDict
from rich.progress import Progress
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import fully_shard
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.datasets import DatasetConfig, parse_dataset
from flow_control.processors.components.vae import VAE, BaseVAE
from flow_control.utils.logging import console, dump_if_failed, get_logger
from flow_control.utils.resize import resize_to_multiple_of
from flow_control.utils.tensor import (
    deep_move_to_device,
    remove_alpha_channel,
    tensor_to_pil,
)
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)

from ..data import DistributedBucketSampler, PaddingAwareDatasetWrapper, collate_fn
from ..mixins import CheckpointingMixin, HsdpMixin, distributed_main
from ..mixins.logging import LoggingMixin
from .convert import convert_to_rgba
from .loss import RGBAVAELoss

logger = get_logger(__name__)


class VaeTrainer(LoggingMixin, HsdpMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")

    # --------------------------------- Model -------------------------------- #
    vae: VAE
    ref_vae: VAE | None = None
    do_convert_to_rgba: bool = True

    # -------------------------------- Dataset -------------------------------- #
    dataset: DatasetConfig
    validation_dataset: DatasetConfig | None = None
    num_dataloader_workers: int = 4
    validation_num_workers: int = 1
    resize_multiple: int = 16
    resize_pixels: int = 0

    # ------------------------------- Training ------------------------------- #
    global_batch_size: int = 8
    train_steps: int = 100000
    checkpoint_steps: int = 5000
    validation_steps: int = 5000
    resume_from_dir: str | None = None
    seed_checkpoint_dir: str | None = None

    # ------------------------------ Optimizers ------------------------------ #
    optimizer_config: OptimizerConfig = {
        "class_name": "AdamW",
        "lr": 1e-5,
        "betas": [0.5, 0.9],
    }
    scheduler_config: SchedulerConfig = {
        "class_name": "CosineAnnealingLR",
        "T_max": 100000,
    }
    disc_optimizer_config: OptimizerConfig = {
        "class_name": "AdamW",
        "lr": 1e-5,
        "betas": [0.5, 0.9],
    }
    disc_scheduler_config: SchedulerConfig = {
        "class_name": "CosineAnnealingLR",
        "T_max": 100000,
    }
    clip_grad_norm: float = 1.0

    # -------------------------------- Losses -------------------------------- #
    kl_scale: float | None = 1e-6
    ref_kl_scale: float | None = None
    lpips_scale: float | None = 0.5
    gan_start_step: int | None = 4000
    generator_loss_weight: float = 1.0
    discriminator_loss_weight: float = 1.0
    use_naive_mse: bool = False
    blend_prob: float = 0.0

    # Weight file paths for LPIPS (required when lpips_scale is not None).
    # Download from:
    #   VGG16:  https://download.pytorch.org/models/vgg16-397923af.pth
    #   LPIPS:  https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1
    vgg16_weights_path: str = "data/vgg16-397923af.pth"
    lpips_weights_path: str = "data/vgg_lpips.pth"

    # ------------------------------ Status bar ------------------------------ #
    _status_fields: dict[str, str] = {
        "train/loss": "Loss: {v:.4f}",
        "train/lr_gen": "LR: {v:.6f}",
    }

    # ------------------------------ Lazy state ------------------------------ #
    _vae_model: Any
    _ref_vae_model: Any = None
    _loss_module: RGBAVAELoss
    _dataloader: StatefulDataLoader
    _validation_dataloader: StatefulDataLoader | None = None
    _optimizer_gen: torch.optim.Optimizer
    _optimizer_disc: torch.optim.Optimizer | None = None
    _scheduler_gen: Any
    _scheduler_disc: Any = None
    _current_step: int = 0

    # ------------------------------ Properties ------------------------------ #

    @property
    def grad_acc_steps(self) -> int:
        return self.global_batch_size // self.world_size

    @property
    def total_epochs(self) -> int:
        steps_per_epoch = len(self._dataloader) // self.grad_acc_steps
        if steps_per_epoch == 0:
            return 1
        return math.ceil(self.train_steps / steps_per_epoch)

    @property
    def current_epoch(self) -> int:
        steps_per_epoch = len(self._dataloader) // self.grad_acc_steps
        if steps_per_epoch == 0:
            return 0
        return self._current_step // steps_per_epoch

    # ------------------------------- Setup ---------------------------------- #

    def _shard_vae(self, model: Any) -> None:
        """Apply FSDP2 sharding to a VAE model.

        Shards encoder, decoder, and any top-level conv modules (quant_conv,
        post_quant_conv) individually so their FSDP forward hooks fire when
        called from ``encode()``/``decode()`` (which bypass ``forward()``).
        """
        fully_shard(model.encoder, mesh=self.mesh)
        fully_shard(model.decoder, mesh=self.mesh)
        # Flux2 VAE has quant_conv / post_quant_conv called inside _encode / _decode
        # (outside of forward), so they need their own FSDP groups.
        for name in ("quant_conv", "post_quant_conv"):
            submodule = getattr(model, name, None)
            if submodule is not None:
                fully_shard(submodule, mesh=self.mesh)
        fully_shard(model, mesh=self.mesh)
        logger.info("Applied FSDP2 sharding to VAE.")

    def load_vae(self) -> None:
        """Load VAE, optionally convert to RGBA, and apply FSDP2.

        When ``seed_checkpoint_dir`` is set, uses the seed checkpoint pattern:
        load on meta → convert architecture → fully_shard → materialize → DCP load.
        This avoids the issue where ``fully_shard`` on a model with loaded weights
        causes some layers to lose their pretrained values.
        """
        vae_loader: BaseVAE[Any] = self.vae

        if self.seed_checkpoint_dir is not None:
            # Seed checkpoint pattern
            model: Any = vae_loader.load_model(torch.device("meta"), frozen=False)

            if self.do_convert_to_rgba:
                model = convert_to_rgba(model)
                vae_loader.model = model

            if self.gradient_checkpointing and hasattr(
                model, "enable_gradient_checkpointing"
            ):
                model.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing on VAE.")

            self._shard_vae(model)

            # Materialize on GPU and load seed checkpoint
            model.to_empty(device=self.device)
            logger.info("VAE materialized on GPU.")

            if not os.path.exists(os.path.join(self.seed_checkpoint_dir, ".metadata")):
                raise ValueError(
                    f"Seed checkpoint directory {self.seed_checkpoint_dir} does not "
                    "exist or is not a valid DCP checkpoint. Set "
                    "launch.generate_dcp_seed = true to auto-generate."
                )
            model_sd, _ = get_state_dict(
                model, [], options=StateDictOptions(strict=False)
            )
            dcp.load(
                model_sd,
                checkpoint_id=self.seed_checkpoint_dir,
                planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
            )
            logger.info(f"Loaded VAE seed checkpoint from {self.seed_checkpoint_dir}.")
        else:
            # Fallback: load directly on device, then shard
            model = vae_loader.load_model(self.device, frozen=False)

            if self.do_convert_to_rgba:
                model = convert_to_rgba(model)
                vae_loader.model = model

            if self.gradient_checkpointing and hasattr(
                model, "enable_gradient_checkpointing"
            ):
                model.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing on VAE.")

            self._shard_vae(model)

        # Train in float32 for stability (model loads as bfloat16 by default)
        model.float()
        self._vae_model = model

        # Load reference VAE (frozen, no FSDP) if needed
        if self.ref_kl_scale is not None and self.ref_vae is not None:
            ref_loader: BaseVAE[Any] = self.ref_vae
            ref_model: Any = ref_loader.load_model(self.device, frozen=True)
            ref_model.float()
            ref_model.eval()
            self._ref_vae_model = ref_model
            logger.info("Loaded frozen reference VAE for KL regularization.")

    def load_loss_module(self) -> None:
        """Initialize the RGBAVAELoss module."""
        self._loss_module = RGBAVAELoss(
            use_lpips=self.lpips_scale is not None,
            use_patchgan=self.gan_start_step is not None,
            use_naive_mse=self.use_naive_mse,
            vgg16_weights_path=self.vgg16_weights_path,
            lpips_weights_path=self.lpips_weights_path,
        )
        # Move frozen parts (LPIPS, buffers) to device
        self._loss_module.to(device=self.device)

        # Wrap discriminator with FSDP2 for gradient sync
        if self.gan_start_step is not None:
            fully_shard(self._loss_module.discriminator, mesh=self.mesh)
            logger.info("Applied FSDP2 sharding to discriminator.")

    def make_optimizer_and_scheduler(self) -> None:
        """Create optimizers for generator (VAE) and discriminator."""
        gen_params = [p for p in self._vae_model.parameters() if p.requires_grad]
        num_gen_params = sum(p.numel() for p in gen_params)
        if num_gen_params == 0:
            raise RuntimeError("No trainable parameters in VAE.")
        self._optimizer_gen = parse_optimizer(self.optimizer_config, gen_params)
        self._scheduler_gen = parse_scheduler(
            self.scheduler_config, self._optimizer_gen
        )
        logger.info(
            f"Created generator optimizer with {num_gen_params / 1e6:.2f}M trainable parameters."
        )

        if self.gan_start_step is not None:
            disc_params = [
                p
                for p in self._loss_module.discriminator.parameters()
                if p.requires_grad
            ]
            self._optimizer_disc = parse_optimizer(
                self.disc_optimizer_config, disc_params
            )
            self._scheduler_disc = parse_scheduler(
                self.disc_scheduler_config, self._optimizer_disc
            )
            logger.info("Created discriminator optimizer.")

    def make_train_dataloader(self) -> None:
        dataset = PaddingAwareDatasetWrapper(parse_dataset(self.dataset))
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            grad_acc_steps=self.grad_acc_steps,
        )
        self._dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=collate_fn,
        )
        logger.info(f"Training dataloader created with {len(dataset)} samples.")

    def make_validation_dataloader(self) -> None:
        if self.validation_dataset is None:
            logger.info("No validation dataset configured, skipping.")
            return
        dataset = PaddingAwareDatasetWrapper(parse_dataset(self.validation_dataset))
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
        )
        logger.info(f"Validation dataloader created with {len(dataset)} samples.")

    # ------------------------------ Checkpointing --------------------------- #

    def state_dict(self) -> dict[str, Any]:
        opts = StateDictOptions(strict=False, ignore_frozen_params=True)
        state: dict[str, Any] = {
            "vae": get_model_state_dict(self._vae_model, options=opts),
            "optimizer_gen": get_optimizer_state_dict(
                self._vae_model, self._optimizer_gen, options=opts
            ),
            "scheduler_gen": self._scheduler_gen.state_dict(),
            "dataloader": self._dataloader.state_dict(),
            "current_step": self._current_step,
        }
        if self._optimizer_disc is not None:
            state["discriminator"] = get_model_state_dict(
                self._loss_module.discriminator, options=opts
            )
            state["optimizer_disc"] = get_optimizer_state_dict(
                self._loss_module.discriminator, self._optimizer_disc, options=opts
            )
            state["scheduler_disc"] = self._scheduler_disc.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        opts = StateDictOptions(strict=False, ignore_frozen_params=True)
        set_model_state_dict(self._vae_model, state_dict["vae"], options=opts)
        set_optimizer_state_dict(
            self._vae_model,
            self._optimizer_gen,
            state_dict["optimizer_gen"],
            options=opts,
        )
        self._scheduler_gen.load_state_dict(state_dict["scheduler_gen"])
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._current_step = state_dict["current_step"]

        if self._optimizer_disc is not None and "discriminator" in state_dict:
            set_model_state_dict(
                self._loss_module.discriminator,
                state_dict["discriminator"],
                options=opts,
            )
            set_optimizer_state_dict(
                self._loss_module.discriminator,
                self._optimizer_disc,
                state_dict["optimizer_disc"],
                options=opts,
            )
            self._scheduler_disc.load_state_dict(state_dict["scheduler_disc"])

    # ------------------------------ Training -------------------------------- #

    def _maybe_blend_to_bg(self, target: torch.Tensor) -> torch.Tensor:
        """With ``blend_prob``, composite RGBA over a random solid background.

        Matches AlphaVAE: each RGB channel of the background is independently
        sampled from {0.0, 0.5, 1.0}, and the result becomes fully opaque.
        """
        if torch.rand(1).item() >= self.blend_prob:
            return target
        rgb = target[:, :3]
        alpha = target[:, 3:4]
        bg = (
            torch.randint(0, 3, (1, 3, 1, 1), device=target.device).to(target.dtype)
            / 2.0
        )
        blended_rgb = rgb * alpha + bg * (1 - alpha)
        return torch.cat([blended_rgb, torch.ones_like(alpha)], dim=1)

    def _prepare_target(self, batch: dict) -> torch.Tensor:
        """Extract clean_image, resize, and normalize to [-1, 1]."""
        target = batch["clean_image"].float()
        target = resize_to_multiple_of(
            target, self.resize_multiple, pixels=self.resize_pixels
        )
        if self.blend_prob > 0:
            target = self._maybe_blend_to_bg(target)
        return target * 2 - 1  # [0,1] -> [-1,1]

    def train_step(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run one VAE generator training micro-step.

        Returns:
            (gen_loss, target, pred_detached, metrics_dict)
        """
        metrics: dict[str, torch.Tensor] = {}
        use_ref_kl = self.ref_kl_scale is not None and self._ref_vae_model is not None

        # -- Compose targets for ref KL --
        target_black: torch.Tensor | None = None
        target_white: torch.Tensor | None = None
        if use_ref_kl:
            fg_alpha = (1 + target[:, 3:]) / 2  # [-1,1] -> [0,1]
            bg_alpha = (1 - target[:, 3:]) / 2
            target_black = target * fg_alpha - bg_alpha
            target_white = target * fg_alpha + bg_alpha
            target_black[:, 3] = 1.0
            target_white[:, 3] = 1.0
            composed = torch.cat((target, target_black, target_white), dim=0)
        else:
            composed = target

        # -- Encode --
        posterior_all: DiagonalGaussianDistribution = self._vae_model.encode(
            composed
        ).latent_dist

        posterior_black: DiagonalGaussianDistribution | None = None
        posterior_white: DiagonalGaussianDistribution | None = None
        ref_posterior_black: DiagonalGaussianDistribution | None = None
        ref_posterior_white: DiagonalGaussianDistribution | None = None

        if use_ref_kl:
            posterior, posterior_black, posterior_white = map(
                DiagonalGaussianDistribution,
                torch.chunk(posterior_all.parameters, 3, dim=0),
            )
            with torch.no_grad():
                assert self._ref_vae_model is not None
                assert target_black is not None and target_white is not None
                # ref_vae is the original 3ch VAE — feed only RGB channels
                # of black/white composites (which have alpha=1, so RGB is
                # the fully-composited image).
                ref_composed = torch.cat(
                    (target_black[:, :3], target_white[:, :3]), dim=0
                )
                ref_posterior_all: DiagonalGaussianDistribution = (
                    self._ref_vae_model.encode(ref_composed).latent_dist
                )
                ref_posterior_black, ref_posterior_white = map(
                    DiagonalGaussianDistribution,
                    torch.chunk(ref_posterior_all.parameters, 2, dim=0),
                )
        else:
            posterior = posterior_all

        # -- Sample and decode --
        vae: Any = self._vae_model  # escape for dynamic attr access
        z = posterior.sample()
        pred = vae.decode(z).sample

        # -- Losses --
        l2_loss = self._loss_module.reconstruction_loss(pred, target)
        loss = l2_loss
        metrics["train/l2_loss"] = l2_loss.detach()

        if self.lpips_scale is not None:
            p_loss = self._loss_module.perceptual_loss(pred, target)
            loss = loss + self.lpips_scale * p_loss
            metrics["train/p_loss"] = p_loss.detach()

        if self.kl_scale is not None:
            kl_norm = self._loss_module.kl_loss(posterior)
            loss = loss + self.kl_scale * kl_norm
            metrics["train/kl_norm"] = kl_norm.detach()

        if use_ref_kl:
            assert posterior_black is not None and ref_posterior_black is not None
            assert posterior_white is not None and ref_posterior_white is not None
            kl_ref_black = self._loss_module.kl_loss(
                posterior_black, ref_posterior_black
            )
            kl_ref_white = self._loss_module.kl_loss(
                posterior_white, ref_posterior_white
            )
            kl_ref = (kl_ref_black + kl_ref_white) / 2
            assert self.ref_kl_scale is not None
            loss = loss + self.ref_kl_scale * kl_ref
            metrics["train/kl_ref"] = kl_ref.detach()

        # -- Generator adversarial loss (after gan_start_step) --
        if (
            self.gan_start_step is not None
            and self._current_step >= self.gan_start_step
        ):
            self._loss_module.requires_grad_(False)
            g_loss = self._loss_module.generator_loss(
                l2_loss,
                pred,
                vae.decoder.conv_out.weight,
            )
            loss = loss + g_loss * self.generator_loss_weight
            metrics["train/g_loss"] = g_loss.detach()

        metrics["train/loss"] = loss.detach()
        return loss, target, pred.detach(), metrics

    def _disc_micro_step(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        is_sync_step: bool,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        """Run discriminator forward+backward for one micro-batch (if active)."""
        if self.gan_start_step is None or self._current_step < self.gan_start_step:
            return
        self._loss_module.discriminator.set_requires_gradient_sync(is_sync_step)  # type: ignore[union-attr]
        d_loss = self._loss_module.discriminator_loss(target, pred)
        d_loss = d_loss * self.discriminator_loss_weight
        d_loss_scaled = d_loss / self.grad_acc_steps
        d_loss_scaled.backward()
        metrics["train/d_loss"] = d_loss.detach()

    def _after_sync_step(self, metrics: dict[str, torch.Tensor]) -> None:
        """Optimizer steps, logging, checkpointing after a gradient sync."""
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self._vae_model.parameters(), self.clip_grad_norm
            )

        self._optimizer_gen.step()
        self._scheduler_gen.step()
        self._optimizer_gen.zero_grad()

        if (
            self._optimizer_disc is not None
            and self.gan_start_step is not None
            and self._current_step >= self.gan_start_step
        ):
            self._optimizer_disc.step()
            self._scheduler_disc.step()
            self._optimizer_disc.zero_grad()

        self._current_step += 1
        metrics["train/lr_gen"] = torch.tensor(
            float(self._scheduler_gen.get_last_lr()[0])
        )
        self.log_metrics(metrics=metrics, step=self._current_step)

        if (self._current_step % self.checkpoint_steps == 0) or (
            self._current_step == self.train_steps
        ):
            self.save(self._current_step)

        if self._current_step % self.validation_steps == 0:
            self.validate_vae(self._current_step)

    def check_loss(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            logger.error(
                f"Non-finite loss detected (loss={loss.item()}). Stopping training."
            )
            raise RuntimeError("Non-finite loss detected.")

    # ------------------------------ Validation ------------------------------ #

    @torch.no_grad()
    def validate_vae(self, step: int) -> None:
        """Encode → decode validation images, log side-by-side comparison."""
        if self._validation_dataloader is None:
            return

        logger.info(f"Validating at step {step}...")
        self._vae_model.eval()

        for batch in self._validation_dataloader:
            batch = deep_move_to_device(batch, self.device)
            target = self._prepare_target(batch)

            pred = self._vae_model.decode(
                self._vae_model.encode(target).latent_dist.sample()
            ).sample

            # Convert back to [0, 1]
            target_01 = (target + 1) / 2
            pred_01 = (pred + 1) / 2

            # Composite over checkerboard for RGBA visualization
            t_rgb = remove_alpha_channel(target_01, "checkerboard")
            p_rgb = remove_alpha_channel(pred_01, "checkerboard")

            # Side by side: input | output
            combined = torch.cat([t_rgb, p_rgb], dim=-1).clamp(0, 1)
            image = tensor_to_pil(combined)
            key = batch.get("__key__", "unknown")
            self.log_image(image, key, step=step, name="vae_validation")

        self._vae_model.train()
        logger.info(f"Completed validation at step {step}.")

    # ------------------------------- Main loop ------------------------------ #

    @distributed_main
    def run(self) -> None:
        self.set_seed()
        self.init_tracker()
        self.load_vae()
        self.load_loss_module()
        self.make_optimizer_and_scheduler()
        self.make_train_dataloader()
        self.make_validation_dataloader()
        os.makedirs(self.checkpoint_root, exist_ok=True)

        if self.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.resume_from_dir)

        self.validate_vae(self._current_step)

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
        )
        task = progress.add_task(
            "Training",
            total=self.train_steps,
            completed=self._current_step,
        )

        with self.status_bar("VAE Training"), progress:
            starting_epoch = self.current_epoch
            for _ in range(starting_epoch, self.total_epochs):
                if hasattr(self._dataloader.sampler, "set_epoch"):
                    self._dataloader.sampler.set_epoch(self.current_epoch)  # type: ignore[union-attr]

                for i, batch in enumerate(self._dataloader):
                    with dump_if_failed(logger, batch):
                        is_sync_step = (i + 1) % self.grad_acc_steps == 0
                        batch = deep_move_to_device(batch, self.device)
                        target = self._prepare_target(batch)

                        # Generator forward + backward
                        self._vae_model.set_requires_gradient_sync(is_sync_step)
                        gen_loss, target_d, pred_d, metrics = self.train_step(target)
                        self.check_loss(gen_loss)
                        (gen_loss / self.grad_acc_steps).backward()

                        # Discriminator forward + backward (if active)
                        self._disc_micro_step(target_d, pred_d, is_sync_step, metrics)

                    if not is_sync_step:
                        continue

                    self._after_sync_step(metrics)
                    progress.advance(task)

                    if self._current_step >= self.train_steps:
                        break

        self.save_dcp_checkpoint(self.get_checkpoint_dir(self._current_step) + "_final")
