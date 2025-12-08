import math

import torch

from flow_control.adapters import ModelAdapter

from .simple_euler import SimpleEulerSampler


class ShiftedEulerSampler(SimpleEulerSampler):
    steps: int = 28
    cfg_scale: float = 1.0

    use_timestep_shift: bool = True
    base_image_seq_len: int = 256
    base_shift: float = 0.5
    max_image_seq_len: int = 4096
    max_shift: float = 1.15
    shift: float = 3.0

    def sample(
        self,
        model: ModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1.0,
        t_end=0.0
    ) -> torch.Tensor:
        self._init_noise_maybe(model, batch, t_start)
        latent_len = model.get_latent_length(batch) # type: ignore
        sigmas = self._make_shifted_sigmas(latent_len, t_start, t_end)
        return self._euler_sample(
            model, batch, sigmas, negative_batch
        )

    def _make_shifted_sigmas(
        self, latent_len: int, t_start=1.0, t_end=0.0
    ) -> torch.Tensor:
        t = torch.linspace(t_start, t_end, self.steps + 1)
        if self.use_timestep_shift:
            m = (self.max_shift - self.base_shift) / (
                self.max_image_seq_len - self.base_image_seq_len
            )
            b = self.base_shift - m * self.base_image_seq_len
            mu = m * latent_len + b
            t = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
        return t
