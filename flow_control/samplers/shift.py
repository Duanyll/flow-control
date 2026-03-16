import math
from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, Tag

from flow_control.adapters.base import Batch


class BaseShift(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    latent_length_from: Literal["actual", "image_size"] = "actual"
    shift_terminal: float | None = None

    def apply(self, sigmas: torch.Tensor, batch: Batch, num_steps: int) -> torch.Tensor:
        seq_len = self._get_seq_len(batch)
        shift_factor = self._calculate_shift_factor(seq_len, num_steps)
        if shift_factor != 1.0:
            sigmas = (shift_factor * sigmas) / (1 + (shift_factor - 1) * sigmas)

        if self.shift_terminal is not None:
            one_minus_z = 1 - sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            sigmas = 1 - (one_minus_z / scale_factor)

        return sigmas

    def _get_seq_len(self, batch: Batch) -> int:
        if self.latent_length_from == "actual":
            return batch["noisy_latents"].shape[1]

        h, w = batch["image_size"]
        return h * w // 256  # assuming patch size 16x16

    @abstractmethod
    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        raise NotImplementedError()


class NoShift(BaseShift):
    type: Literal["none"] = "none"

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        return 1.0


class ConstantShift(BaseShift):
    type: Literal["constant"] = "constant"
    shift_value: float = 1.0

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        return self.shift_value


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


class SquaredShift(BaseShift):
    type: Literal["squared"] = "squared"
    base_image_seq_len: int = 256

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        return (seq_len / self.base_image_seq_len) ** 0.5


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
    Annotated[NoShift, Tag("none")]
    | Annotated[ConstantShift, Tag("constant")]
    | Annotated[LinearShift, Tag("linear")]
    | Annotated[SquaredShift, Tag("squared")]
    | Annotated[Flux2Shift, Tag("flux2")],
    Discriminator("type"),
]
