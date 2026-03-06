from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.datasets import parse_dataset
from flow_control.rewards import Reward
from flow_control.samplers.euler import SdeTrajectory
from flow_control.utils.common import (
    deep_cast_float_dtype,
    deep_move_to_device,
)
from flow_control.utils.ema import apply_ema_maybe, apply_init_maybe
from flow_control.utils.logging import console, get_logger

from .advantage import Advantage, PerPromptAdvantage
from .data import DistributedKRepeatSampler, PaddingAwareDatasetWrapper, collate_fn
from .hsdp_engine import DistributedExitSignal, distributed_main
from .trainer_base import HsdpTrainerBase, HsdpTrainerBaseConfig

logger = get_logger(__name__)


class HsdpGrpoTrainerConfig(HsdpTrainerBaseConfig):
    reward: Reward

    # GRPO hyperparameters
    num_samples_per_prompt: int = 4
    num_batches_per_epoch: int = 2
    num_inner_epochs: int = 1
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    kl_beta: float = 0.0
    timestep_fraction: float = 1.0
    advantage: Advantage = PerPromptAdvantage()

    # Override defaults from base
    optimizer: dict[str, Any] = {"class_name": "AdamW", "lr": 3e-4}
    global_batch_size: int = 4
    ema_decay: float = 1.0
    sample_steps: int = 20
    num_dataloader_workers: int = 0

    train_batch_size: int | None = None


class HsdpGrpoTrainer(HsdpTrainerBase[HsdpGrpoTrainerConfig]):
    conf: HsdpGrpoTrainerConfig

    current_epoch: int = 0

    @property
    def batch_per_rank(self) -> int:
        return self.conf.global_batch_size // self.world_size

    def __init__(self, **kwargs):
        self.conf = HsdpGrpoTrainerConfig(**kwargs)
        super().__init__(**kwargs)

    def make_optimizer_and_scheduler(self, enable_init_backup: bool = False):
        need_init = self.conf.kl_beta > 0 and self.model.peft_lora_rank == 0
        super().make_optimizer_and_scheduler(enable_init_backup=need_init)
        if need_init:
            logger.info("Init backup enabled for reference model (kl_beta > 0).")

    def make_train_dataloader(self):
        dataset = PaddingAwareDatasetWrapper(parse_dataset(self.conf.dataset))
        sampler = DistributedKRepeatSampler(
            dataset=dataset,
            batch_per_rank=self.batch_per_rank,
            k=self.conf.num_samples_per_prompt,
            num_batches=self.conf.num_batches_per_epoch,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self.conf.seed,
        )
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.conf.num_dataloader_workers,
            collate_fn=collate_fn,
        )

    def _save_extra_state(self, state: dict) -> None:
        state["current_epoch"] = self.current_epoch

    def _load_extra_state(self, state_dict: dict) -> None:
        self.current_epoch = state_dict.get("current_epoch", 0)

    @contextmanager
    def reference_model(self):
        """Temporarily switch to reference model weights."""
        if self.model.peft_lora_rank > 0:
            with self.transformer.disable_adapter():
                yield
        else:
            with apply_init_maybe(self.optimizer):
                yield

    # --- GRPO loss ---

    @staticmethod
    def grpo_loss(
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        clip_range: float,
        adv_clip_max: float,
        mean: torch.Tensor | None = None,
        ref_mean: torch.Tensor | None = None,
        std_dev: torch.Tensor | None = None,
        kl_beta: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute GRPO loss.

        L = E[max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))] + beta * KL
        """
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
        ratio = torch.exp(log_prob - old_log_prob)

        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        metrics: dict[str, torch.Tensor] = {
            "train/policy_loss": policy_loss.detach(),
            "train/ratio_mean": ratio.detach().mean(),
            "train/approx_kl": (
                0.5 * torch.mean((log_prob - old_log_prob) ** 2)
            ).detach(),
            "train/clipfrac": (
                torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
            ).detach(),
        }

        loss = policy_loss

        if (
            kl_beta > 0
            and mean is not None
            and ref_mean is not None
            and std_dev is not None
        ):
            kl_loss = ((mean - ref_mean) ** 2).mean(dim=tuple(range(1, mean.ndim))) / (
                2 * std_dev**2
            )
            kl_loss = torch.mean(kl_loss)
            loss = loss + kl_beta * kl_loss
            metrics["train/kl_loss"] = kl_loss.detach()

        metrics["train/loss"] = loss.detach()
        return loss, metrics

    # --- Core training phases ---

    def _collect_samples(self) -> list[dict[str, Any]]:
        """Sampling phase: generate images and compute rewards."""
        samples: list[dict[str, Any]] = []
        self.transformer.eval()

        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= self.conf.num_batches_per_epoch * self.batch_per_rank:
                break

            batch = deep_move_to_device(batch, self.device)
            batch = deep_cast_float_dtype(batch, self.model.dtype)

            # Initialize noise
            generator = torch.Generator(device=self.device).manual_seed(
                self.conf.seed + self.current_epoch * 10000 + batch_idx
            )
            self.processor.initialize_latents(
                batch,
                generator=generator,
                device=self.device,
                dtype=self.model.dtype,
            )

            negative_batch: Any = (
                self.processor.get_negative_batch(batch)
                if self.sampler.cfg_scale > 1.0
                else None
            )

            # SDE sample with log_prob
            with torch.no_grad():
                trajectory: SdeTrajectory = self.sampler.sample_with_logprob(
                    self.model, batch, negative_batch=negative_batch
                )

            # Decode final latents to images and merge into batch
            final_latents = trajectory["latents"][:, -1]
            decoded = self.processor.decode_output(final_latents, batch)
            batch.update(decoded)

            # Compute reward — score() receives the full batch dict
            with torch.no_grad():
                reward = self.conf.reward.score(batch)

            samples.append(
                {
                    "trajectory": trajectory,
                    "reward": reward.view(1) if reward.ndim == 0 else reward,
                    "key": batch.get("__key__", "unknown"),
                    "batch": batch,
                    "negative_batch": negative_batch,
                }
            )

        return samples

    def _compute_advantages(self, samples: list[dict[str, Any]]) -> torch.Tensor:
        """Compute advantages using gathered rewards across all GPUs.

        Uses the ``__key__`` field from each sample's batch to group samples
        that belong to the same prompt for per-prompt advantage estimation.
        """
        local_rewards = torch.cat([s["reward"] for s in samples], dim=0)

        # Gather rewards across all processes
        gathered_rewards_list: list[torch.Tensor] = [
            torch.zeros_like(local_rewards) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_rewards_list, local_rewards)
        gathered_rewards = torch.cat(gathered_rewards_list, dim=0)

        # Gather keys across processes to identify prompt groups
        local_keys = [s["key"] for s in samples]
        all_keys_nested: list[Any] = [None] * self.world_size
        dist.all_gather_object(all_keys_nested, local_keys)
        all_keys: list[str] = []
        for key_list in all_keys_nested:
            all_keys.extend(key_list)

        # Map keys to integer IDs for the advantage estimator
        unique_keys = list(dict.fromkeys(all_keys))
        key_to_id = {k: i for i, k in enumerate(unique_keys)}
        prompt_ids = torch.tensor(
            [key_to_id[k] for k in all_keys],
            device=gathered_rewards.device,
        )

        num_timesteps = samples[0]["trajectory"]["log_probs"].shape[1]

        # Compute advantages [global_B, T]
        advantages = self.conf.advantage.compute(
            gathered_rewards, prompt_ids, num_timesteps
        )

        # Log reward stats
        reward_mean = gathered_rewards.mean().item()
        reward_std = gathered_rewards.std().item()
        self.log_metrics(
            {
                "reward/mean": reward_mean,
                "reward/std": reward_std,
                "reward/adv_abs_mean": advantages.abs().mean().item(),
            }
        )
        logger.info(
            f"Epoch {self.current_epoch}: reward_mean={reward_mean:.4f}, "
            f"reward_std={reward_std:.4f}"
        )

        # Ungather: keep only this rank's slice
        local_b = local_rewards.shape[0]
        start = self.rank * local_b
        end = start + local_b
        return advantages[start:end]

    def _train_on_samples(
        self,
        samples: list[dict[str, Any]],
        advantages: torch.Tensor,
    ):
        """Training phase: update model using collected samples and advantages."""
        self.transformer.train()

        num_timesteps = samples[0]["trajectory"]["log_probs"].shape[1]
        num_train_timesteps = max(1, int(num_timesteps * self.conf.timestep_fraction))

        for _inner_epoch in range(self.conf.num_inner_epochs):
            perm = torch.randperm(len(samples))
            shuffled_samples = [samples[perm[i]] for i in range(len(samples))]
            shuffled_advantages = advantages[perm]

            for sample_idx, sample in enumerate(shuffled_samples):
                trajectory = sample["trajectory"]
                batch = sample["batch"]
                negative_batch = sample["negative_batch"]
                sample_advantages = shuffled_advantages[sample_idx]  # [T]
                old_log_probs = trajectory["log_probs"][0]  # [T]
                sigmas = trajectory["timesteps"]  # [T+1]

                if self.conf.timestep_fraction < 1.0:
                    step_indices = sorted(
                        torch.randperm(num_timesteps)[:num_train_timesteps].tolist()
                    )
                else:
                    step_indices = list(range(num_timesteps))

                metrics: dict[str, torch.Tensor] = {}
                for j in step_indices:
                    latent_t = trajectory["latents"][:, j]
                    latent_next = trajectory["latents"][:, j + 1]
                    sigma = sigmas[j : j + 1]
                    sigma_next = sigmas[j + 1 : j + 2]

                    log_prob, mean, std_dev = self.sampler.compute_logprob_at_step(
                        self.model,
                        batch,
                        latent_t,
                        latent_next,
                        sigma,
                        sigma_next,
                        negative_batch=negative_batch,
                    )

                    ref_mean = None
                    if self.conf.kl_beta > 0:
                        with torch.no_grad(), self.reference_model():
                            _, ref_mean, _ = self.sampler.compute_logprob_at_step(
                                self.model,
                                batch,
                                latent_t,
                                latent_next,
                                sigma,
                                sigma_next,
                                negative_batch=negative_batch,
                            )

                    loss, metrics = self.grpo_loss(
                        log_prob=log_prob,
                        old_log_prob=old_log_probs[j : j + 1],
                        advantages=sample_advantages[j : j + 1],
                        clip_range=self.conf.clip_range,
                        adv_clip_max=self.conf.adv_clip_max,
                        mean=mean,
                        ref_mean=ref_mean,
                        std_dev=std_dev,
                        kl_beta=self.conf.kl_beta,
                    )

                    loss.backward()

                # Step optimizer after all timesteps for this sample
                if self.conf.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.transformer.parameters(), self.conf.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.current_step += 1

                # Gather and log metrics
                all_metrics: dict[str, float] = {}
                for key, value in metrics.items():
                    values_list: list[Any] = [None] * self.world_size
                    dist.all_gather_object(values_list, value.item())
                    all_metrics[key] = sum(
                        v for v in values_list if v is not None
                    ) / len(values_list)
                self.log_metrics(all_metrics)

    # --- Main loop ---

    @distributed_main
    def run(self):
        self.set_seed()
        self.init_tracker()
        self.load_transformer(self.model, self.conf.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.make_train_dataloader()
        self.make_sample_dataloader_maybe()

        # Load processor for VAE decoding
        self.processor.load_models("decode", device=self.device)

        # Load reward model
        self.conf.reward.load_model(self.device)

        if self.conf.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.conf.resume_from_dir)

        self.sample_and_log()

        console.rule("[bold blue]Starting GRPO training[/bold blue]")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" Epoch: {task.fields[epoch]}"),
            TextColumn(" Step: {task.fields[step]}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        task = progress.add_task(
            "GRPO Training...",
            total=self.conf.train_steps,
            completed=self.current_step,
            epoch=self.current_epoch,
            step=self.current_step,
        )

        with progress, DistributedExitSignal(self) as exit_signal:
            while self.current_step < self.conf.train_steps:
                if hasattr(self.dataloader.sampler, "set_epoch"):
                    self.dataloader.sampler.set_epoch(self.current_epoch)  # type: ignore[union-attr]

                # Sampling phase
                logger.info(f"Epoch {self.current_epoch}: starting sampling phase...")
                samples = self._collect_samples()

                # Advantage computation
                advantages = self._compute_advantages(samples)

                # Training phase
                logger.info(f"Epoch {self.current_epoch}: starting training phase...")
                self._train_on_samples(samples, advantages)

                progress.update(
                    task,
                    completed=self.current_step,
                    epoch=self.current_epoch,
                    step=self.current_step,
                )

                del samples, advantages
                torch.cuda.empty_cache()

                self.current_epoch += 1

                if exit_signal:
                    logger.info(
                        "Exit signal received. Saving checkpoint and exiting..."
                    )
                    self.save_dcp_checkpoint(self.get_checkpoint_dir(self.current_step))
                    return

                if self.current_step % self.conf.checkpoint_steps == 0:
                    self.save_dcp_checkpoint(self.get_checkpoint_dir(self.current_step))
                    self.rotate_checkpoints_maybe()

                if self.current_epoch % self.conf.sample_steps == 0:
                    self.sample_and_log()

        with apply_ema_maybe(self.optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self.current_step) + "_final"
            )

        console.rule("[bold green]GRPO Training completed[/bold green]")
