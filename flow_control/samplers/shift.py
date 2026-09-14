import math
from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.adapters.base import Batch
from flow_control.utils.registry import Registry, RegistryUnion


class BaseShift(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    image_seq_len_from: Literal["actual", "image_size"] = "actual"
    """Image sequence length for resolution-dependent shift: ``actual`` reads it
    off ``noisy_latents`` (scaled to ``model_image_size``), ``image_size`` assumes
    16x16 pixels per token."""
    shift_terminal: float | None = None

    def apply(self, sigmas: torch.Tensor, row: Batch, num_steps: int) -> torch.Tensor:
        shift_factor = self._shift_factor(row, num_steps)
        if shift_factor != 1.0:
            sigmas = (shift_factor * sigmas) / (1 + (shift_factor - 1) * sigmas)

        if self.shift_terminal is not None:
            one_minus_z = 1 - sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            sigmas = 1 - (one_minus_z / scale_factor)

        return sigmas

    def _shift_factor(self, row: Batch, num_steps: int) -> float:
        return self._calculate_shift_factor(self._get_seq_len(row), num_steps)

    def _get_seq_len(self, row: Batch) -> int:
        # The model may see less than the whole image per forward (tiles).
        height, width = row["image_size"]
        model_h, model_w = cast(dict[str, Any], row).get(
            "model_image_size", (height, width)
        )
        if self.image_seq_len_from == "actual":
            return row["noisy_latents"].shape[1] * model_h * model_w // (height * width)
        return model_h * model_w // 256  # assuming patch size 16x16

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
