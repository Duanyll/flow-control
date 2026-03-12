import math
import random
from typing import Literal, NotRequired, TypedDict

import torch

from flow_control.adapters.base import BaseModelAdapter, Batch

from .base import BaseSampler, make_sample_progress


class SdeTrajectory(TypedDict):
    latents: torch.Tensor  # [B, T+1, N, D] selected training-window latents
    log_probs: torch.Tensor  # [B, T] per-step log probabilities
    timesteps: torch.Tensor  # [T+1] sigma sequence for selected training window
    final_latents: NotRequired[torch.Tensor]  # [B, N, D] final latents for decoding


class EulerSampler(BaseSampler):
    type: Literal["euler"] = "euler"
    steps: int = 50
    noise_level: float = 0.0  # 0.0 = pure ODE (default), >0 = SDE
    sde_type: Literal["sde", "cps"] = "sde"
    sde_window_size: int | None = None
    sde_window_range: tuple[int, int] | None = None

    def _select_training_window(self, num_timesteps: int) -> tuple[int, int]:
        if self.sde_window_size:
            range_start, range_end = self.sde_window_range or (0, num_timesteps - 1)
            assert 0 <= range_start < range_end < num_timesteps, (
                "Invalid sde_window_range"
            )
            window_start = random.randint(range_start, range_end - self.sde_window_size)
            return window_start, window_start + self.sde_window_size
        else:
            return 0, num_timesteps - 1

    def _sample(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start: float = 1.0,
        t_end: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor | SdeTrajectory:
        sigmas = torch.linspace(t_start, t_end, self.steps + 1)
        return self._euler_sample(
            model,
            batch,
            sigmas,
            negative_batch,
            return_trajectory=return_trajectory,
        )

    def _euler_sample(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        sigmas: torch.Tensor,
        negative_batch: Batch | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | SdeTrajectory:
        device = model.device
        dtype = model.dtype

        sigmas = sigmas.to(device=device, dtype=torch.float32)
        latents = batch["noisy_latents"].float()

        train_start, train_end = self._select_training_window(self.steps)
        selected_latents: list[torch.Tensor] | None = [] if return_trajectory else None
        all_log_probs: list[torch.Tensor] = []

        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=self.steps)

            for i in range(self.steps):
                t_cur = sigmas[i : i + 1]
                t_next = sigmas[i + 1 : i + 2]
                velocity = self.get_guided_velocity(
                    model, latents, t_cur, batch, negative_batch
                )
                step_noise_level = (
                    self.noise_level if train_start <= i < train_end else 0.0
                )
                if return_trajectory and i == train_start:
                    assert selected_latents is not None
                    selected_latents.append(latents)
                latents, log_prob, mean, std_dev = self.sde_step(
                    velocity,
                    latents,
                    t_cur,
                    t_next,
                    noise_level=step_noise_level,
                )

                if return_trajectory and train_start <= i < train_end:
                    assert selected_latents is not None
                    selected_latents.append(latents)
                    all_log_probs.append(log_prob)

                progress.advance(task)

        if not return_trajectory:
            return latents.to(dtype)

        assert selected_latents is not None
        stacked_latents = torch.stack(selected_latents, dim=1)
        stacked_log_probs = torch.stack(all_log_probs, dim=1)

        return SdeTrajectory(
            latents=stacked_latents,
            log_probs=stacked_log_probs,
            timesteps=sigmas[train_start : train_end + 1],
            final_latents=latents.to(dtype),
        )

    def sde_step(
        self,
        velocity: torch.Tensor,
        latent: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        noise_level: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single SDE step. Returns (next_latent, log_prob, mean, std_dev).

        All computation is performed in float32 to avoid bf16 overflow.
        When noise_level=0, degenerates to standard ODE (no noise, log_prob=0).
        """
        velocity = velocity
        latent = latent
        if prev_sample is not None:
            prev_sample = prev_sample
        step_noise_level = self.noise_level if noise_level is None else noise_level

        dt = sigma_next - sigma  # negative (sigma decreases over time)

        if step_noise_level == 0.0:
            # Pure ODE step
            next_latent = latent + velocity * dt
            log_prob = torch.zeros(latent.shape[0], device=latent.device)
            mean = next_latent
            std_dev = torch.tensor(0.0, device=latent.device)
            return next_latent, log_prob, mean, std_dev

        if self.sde_type == "sde":
            # Match flow_grpo: only guard the denominator when sigma == 1.
            sigma_denom = torch.where(
                sigma == 1.0,
                sigma_next,
                sigma,
            )
            std_dev_t = torch.sqrt(sigma / (1 - sigma_denom)) * step_noise_level

            # SDE mean update
            mean = (
                latent * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + velocity * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            noise_scale = std_dev_t * torch.sqrt(-dt)

            if prev_sample is None:
                noise = torch.randn_like(latent)
                next_latent = mean + noise_scale * noise
            else:
                next_latent = prev_sample

            log_prob = (
                -((next_latent.detach() - mean) ** 2) / (2 * noise_scale**2)
                - torch.log(noise_scale)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif self.sde_type == "cps":
            std_dev_t = sigma_next * math.sin(step_noise_level * math.pi / 2)
            pred_original_sample = latent - sigma * velocity
            noise_estimate = latent + velocity * (1 - sigma)
            mean = pred_original_sample * (
                1 - sigma_next
            ) + noise_estimate * torch.sqrt(sigma_next**2 - std_dev_t**2)

            if prev_sample is None:
                noise = torch.randn_like(latent)
                next_latent = mean + std_dev_t * noise
            else:
                next_latent = prev_sample

            # Remove all constants (as in reference implementation)
            log_prob = -((next_latent.detach() - mean) ** 2)
        else:
            msg = f"Unknown sde_type: {self.sde_type}"
            raise ValueError(msg)

        # Mean over all dims except batch
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        # Return scalar std_dev (squeeze spatial dims)
        if isinstance(std_dev_t, torch.Tensor) and std_dev_t.ndim > 0:
            std_dev_scalar = std_dev_t.squeeze()
        else:
            std_dev_scalar = std_dev_t

        return next_latent, log_prob, mean, std_dev_scalar

    def compute_logprob_at_step(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        latent_t: torch.Tensor,
        latent_t_minus_1: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        negative_batch: Batch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given known trajectory, compute log_prob under current policy.

        Runs a single forward pass + sde_step with prev_sample set to get
        log_prob of the known transition.

        Returns:
            (log_prob, mean, std_dev) tuple.
        """
        velocity = self.get_guided_velocity(
            model, latent_t, sigma, batch, negative_batch
        )

        _, log_prob, mean, std_dev = self.sde_step(
            velocity, latent_t, sigma, sigma_next, prev_sample=latent_t_minus_1
        )

        return log_prob, mean, std_dev
