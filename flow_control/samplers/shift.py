import math
from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.adapters.base import Batch
from flow_control.utils.registry import Registry, RegistryUnion


class BaseShift(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    latent_length_from: Literal["actual", "image_size"] = "actual"
    shift_terminal: float | None = None

    def apply(self, sigmas: torch.Tensor, batch: Batch, num_steps: int) -> torch.Tensor:
        shift_factor = self._shift_factor(batch, num_steps)
        if shift_factor != 1.0:
            sigmas = (shift_factor * sigmas) / (1 + (shift_factor - 1) * sigmas)

        if self.shift_terminal is not None:
            one_minus_z = 1 - sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            sigmas = 1 - (one_minus_z / scale_factor)

        return sigmas

    def inverse_sigma(
        self,
        value: float,
        batch: Batch,
        num_steps: int,
        t_end: float = 0.0,
    ) -> float:
        """Map an actual (shifted) sigma back to its canonical time.

        Inverse of :meth:`apply` for a full canonical grid ending at ``t_end``,
        using the same resolution parameters. The ``shift_terminal`` affine
        post-pass is inverted first (reverse order of ``apply``); its scale
        depends on the grid's terminal value, recomputed here as the pointwise
        shift of ``t_end`` so callers need not pass a grid. Then the pointwise
        ``y = a*x / (1 + (a-1)*x)`` inverts analytically as
        ``x = y / (a - (a-1)*y)``.
        """
        shift_factor = self._shift_factor(batch, num_steps)
        if self.shift_terminal is not None:
            terminal = t_end
            if shift_factor != 1.0:
                terminal = (shift_factor * t_end) / (1 + (shift_factor - 1) * t_end)
            scale_factor = (1 - terminal) / (1 - self.shift_terminal)
            value = 1 - (1 - value) * scale_factor
        if shift_factor != 1.0:
            value = value / (shift_factor - (shift_factor - 1) * value)
        return value

    def _shift_factor(self, batch: Batch, num_steps: int) -> float:
        return self._calculate_shift_factor(self._get_seq_len(batch), num_steps)

    def _get_seq_len(self, batch: Batch) -> int:
        if self.latent_length_from == "actual":
            return batch["noisy_latents"].shape[1]

        h, w = batch["image_size"]
        return h * w // 256  # assuming patch size 16x16

    @abstractmethod
    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        raise NotImplementedError()


shift_registry: Registry[BaseShift] = Registry("shift", base=BaseShift)


@shift_registry.register("constant")
class ConstantShift(BaseShift):
    """Resolution-independent shift factor; the default 1.0 is no shift."""

    type: Literal["constant"] = "constant"
    shift_value: float = 1.0

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        return self.shift_value


@shift_registry.register("linear")
class LinearShift(BaseShift):
    type: Literal["linear"] = "linear"
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        m = (self.max_shift - self.base_shift) / (
            self.max_image_seq_len - self.base_image_seq_len
        )
        b = self.base_shift - m * self.base_image_seq_len
        mu = seq_len * m + b
        return math.exp(mu)


@shift_registry.register("squared")
class SquaredShift(BaseShift):
    type: Literal["squared"] = "squared"
    base_image_seq_len: int = 256

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        return (seq_len / self.base_image_seq_len) ** 0.5


@shift_registry.register("flux2")
class Flux2Shift(BaseShift):
    type: Literal["flux2"] = "flux2"
    a1: float = 8.73809524e-05
    b1: float = 1.89833333
    a2: float = 0.00016927
    b2: float = 0.45666666
    image_seq_len_threshold: int = 4300
    c: float = 190.0
    d: float = 200.0

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        if seq_len > self.image_seq_len_threshold:
            return self.a2 * seq_len + self.b2

        m_200 = self.a2 * seq_len + self.b2
        m_10 = self.a1 * seq_len + self.b1
        a = (m_200 - m_10) / self.c
        b = m_200 - self.d * a
        return a * num_steps + b


Shift = Annotated[
    BaseShift,
    # A bare number is a constant shift factor: ``"shift": 3.0``.
    RegistryUnion(shift_registry, "type", number_as=("constant", "shift_value")),
]
