import math
from typing import Literal

import torch

from flow_control.adapters import Batch, ModelAdapter

from .euler import EulerSampler, SdeTrajectory


class ShiftedEulerSampler(EulerSampler):
    latent_length_from: Literal["actual", "image_size"] = "actual"
    shift_terminal: float | None = None

    def _calculate_shift_factor(self, seq_len: int) -> float:
        raise NotImplementedError("Subclasses must implement _calculate_shift_factor")

    def _make_shifted_sigmas(
        self, seq_len: int, t_start: float, t_end: float
    ) -> torch.Tensor:
        t = torch.linspace(t_start, t_end, self.steps + 1)

        shift_factor = self._calculate_shift_factor(seq_len)
        if shift_factor != 1.0:
            t = (shift_factor * t) / (1 + (shift_factor - 1) * t)

        if self.shift_terminal is not None:
            one_minus_z = 1 - t
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            t = 1 - (one_minus_z / scale_factor)

        return t

    def _sample(
        self,
        model: ModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ) -> torch.Tensor:
        if self.latent_length_from == "actual":
            seq_len = batch["noisy_latents"].shape[1]
        else:
            h, w = batch["image_size"]
            seq_len = h * w // 256  # assuming patch size 16x16

        sigmas = self._make_shifted_sigmas(seq_len, t_start, t_end)
        return self._euler_sample(model, batch, sigmas, negative_batch)

    def sample_with_logprob(
        self,
        model: ModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
    ) -> SdeTrajectory:
        """SDE sampling with log probability tracking. Uses shifted sigmas."""
        if self.latent_length_from == "actual":
            seq_len = batch["noisy_latents"].shape[1]
        else:
            h, w = batch["image_size"]
            seq_len = h * w // 256  # assuming patch size 16x16

        sigmas = self._make_shifted_sigmas(seq_len, t_start=1.0, t_end=0.0)
        return self._euler_sample_with_logprob(model, batch, sigmas, negative_batch)


class ConstantShiftSampler(ShiftedEulerSampler):
    shift_value: float = 1.0

    def _calculate_shift_factor(self, seq_len: int) -> float:
        return self.shift_value


class LinearShiftSampler(ShiftedEulerSampler):
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15

    def _calculate_shift_factor(self, seq_len: int) -> float:
        m = (self.max_shift - self.base_shift) / (
            self.max_image_seq_len - self.base_image_seq_len
        )
        b = self.base_shift - m * self.base_image_seq_len
        mu = seq_len * m + b
        return math.exp(mu)


class SquaredShiftSampler(ShiftedEulerSampler):
    base_image_seq_len: int = 256

    def _calculate_shift_factor(self, seq_len: int) -> float:
        mu = (seq_len / self.base_image_seq_len) ** 0.5
        return mu


class Flux2ShiftSampler(ShiftedEulerSampler):
    a1: float = 8.73809524e-05
    b1: float = 1.89833333
    a2: float = 0.00016927
    b2: float = 0.45666666
    image_seq_len_threshold: int = 4300
    c: float = 190.0
    d: float = 200.0

    def _calculate_shift_factor(self, seq_len: int) -> float:
        if seq_len > self.image_seq_len_threshold:
            return self.a2 * seq_len + self.b2

        m_200 = self.a2 * seq_len + self.b2
        m_10 = self.a1 * seq_len + self.b1
        a = (m_200 - m_10) / self.c
        b = m_200 - self.d * a
        mu = a * self.steps + b
        return mu
