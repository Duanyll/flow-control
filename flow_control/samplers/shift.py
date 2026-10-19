"""Sigma-grid shifts as invertible curves σ = F(u) on [0, 1]."""

import math
from typing import Annotated, Literal, Protocol

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry, RegistryUnion

from .node import Row


class Curve(Protocol):
    """σ = F(u) over a whole u grid, endpoints included; ``inverse`` is f's alone."""

    def __call__(self, u: torch.Tensor) -> torch.Tensor: ...
    def inverse(self, sigma: float) -> float: ...


class ShiftCurve:
    """f(u) = s·u / (1 + (s − 1)·u); a grid ending at 0 is stretched so its penultimate σ is ``terminal``."""

    def __init__(self, factor: float, terminal: float | None) -> None:
        self.factor, self.terminal = factor, terminal

    def f(self, u: torch.Tensor) -> torch.Tensor:
        s = self.factor
        return (s * u) / (1 + (s - 1) * u)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        sigma = self.f(u)
        if self.terminal is None or len(u) < 3 or u[-1] != 0:
            return sigma
        scale = (1 - sigma[-2]) / (1 - self.terminal)
        return torch.where(sigma > 0, 1 - (1 - sigma) / scale, sigma)

    def inverse(self, sigma: float) -> float:
        return sigma / (sigma + self.factor * (1 - sigma))


class DiffusersCurve(ShiftCurve):
    """diffusers' set_timesteps grid: f over linspace(f(1), f(1/N), n), then 0; full 1 -> 0 grids only."""

    def __init__(self, factor: float, train_steps: int) -> None:
        super().__init__(factor, None)
        self.a = float(self.f(torch.tensor(1 / train_steps)))

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if u[0] != 1 or u[-1] != 0:
            raise ValueError(
                "diffusers_grid is defined on a full 1 -> 0 grid; a partial grid "
                "needs custom_sigmas."
            )
        n, a = len(u) - 1, self.a
        inner = a + (1 - a) * (u * n - 1) / (n - 1)
        return torch.where(u * n >= 1, self.f(inner), torch.zeros_like(u))


class BaseShift(BaseModel):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    image_seq_len_from: Literal["actual", "image_size"] = "actual"
    """``actual`` reads the token count off ``noisy_latents``; ``image_size`` assumes 16x16 px per token."""
    shift_terminal: float | None = None
    """Stretch the final positive sigma of the full grid to this value."""
    diffusers_grid: int | None = None
    """diffusers' ``set_timesteps`` grid with this many training timesteps."""

    def curve(self, row: Row, num_steps: int) -> Curve:
        factor = self.factor(self._seq_len(row), num_steps)
        if self.diffusers_grid:
            return DiffusersCurve(factor, self.diffusers_grid)
        return ShiftCurve(factor, self.shift_terminal or None)

    def _seq_len(self, row: Row) -> int:
        # The model may see less than the whole image per forward (tiles).
        height, width = row["image_size"]
        model_h, model_w = row.get("model_image_size", (height, width))
        if self.image_seq_len_from == "actual":
            return row["noisy_latents"].shape[1] * model_h * model_w // (height * width)
        return model_h * model_w // 256

    def factor(self, seq_len: int, num_steps: int) -> float:
        raise NotImplementedError


shift_registry: Registry[BaseShift] = Registry("shift", base=BaseShift)


@shift_registry.register("constant")
class ConstantShift(BaseShift):
    """Resolution-independent shift factor; the default 1.0 is no shift."""

    type: Literal["constant"] = "constant"
    shift_value: float = 1.0

    def factor(self, seq_len: int, num_steps: int) -> float:
        return self.shift_value


@shift_registry.register("linear")
class LinearShift(BaseShift):
    type: Literal["linear"] = "linear"
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15

    def factor(self, seq_len: int, num_steps: int) -> float:
        m = (self.max_shift - self.base_shift) / (
            self.max_image_seq_len - self.base_image_seq_len
        )
        b = self.base_shift - m * self.base_image_seq_len
        return math.exp(seq_len * m + b)


@shift_registry.register("squared")
class SquaredShift(BaseShift):
    type: Literal["squared"] = "squared"
    base_image_seq_len: int = 256

    def factor(self, seq_len: int, num_steps: int) -> float:
        return (seq_len / self.base_image_seq_len) ** 0.5


@shift_registry.register("flux2")
class Flux2Shift(BaseShift):
    """FLUX.2's empirical ``mu`` (diffusers ``compute_empirical_mu``) as the factor ``e^mu``."""

    type: Literal["flux2"] = "flux2"
    a1: float = 8.73809524e-05
    b1: float = 1.89833333
    a2: float = 0.00016927
    b2: float = 0.45666666
    image_seq_len_threshold: int = 4300
    c: float = 190.0
    d: float = 200.0

    def factor(self, seq_len: int, num_steps: int) -> float:
        m_200 = self.a2 * seq_len + self.b2
        if seq_len > self.image_seq_len_threshold:
            return math.exp(m_200)
        m_10 = self.a1 * seq_len + self.b1
        a = (m_200 - m_10) / self.c
        return math.exp(a * num_steps + m_200 - self.d * a)


Shift = Annotated[
    BaseShift,
    # A bare number is a constant shift factor: ``"shift": 3.0``.
    RegistryUnion(shift_registry, "type", number_as=("constant", "shift_value")),
]


if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print

    # Golden grids from diffusers' FlowMatchEulerDiscreteScheduler with
    # ``time_shift_type="exponential"`` and ``mu = compute_empirical_mu(...)``
    # (the FLUX.2 Klein scheduler config), sigmas = linspace(1, 1/steps, steps).
    golden = {
        (1024, 20): (6.796091, [1.0, 0.992315, 0.983914, 0.974691], 0.263454),
        (1024, 28): (6.418449, [1.0, 0.994263, 0.988157, 0.981647], 0.192063),
        (4096, 50): (7.563629, [1.0, 0.997309, 0.994521, 0.991632], 0.133719),
    }
    shift = Flux2Shift()
    for (seq_len, steps), (factor, head, last) in golden.items():
        assert abs(shift.factor(seq_len, steps) - factor) < 1e-5
        side = 16 * int(seq_len**0.5)
        row = {"image_size": (side, side), "noisy_latents": torch.zeros(1, seq_len, 1)}
        curve = shift.curve(row, steps)
        shifted = curve(torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64))
        assert torch.allclose(
            shifted[:4], torch.tensor(head, dtype=torch.float64), atol=2e-6
        ), (seq_len, steps, shifted[:4])
        assert abs(shifted[-1].item() - last) < 2e-6, (seq_len, steps, shifted[-1])
        assert abs(curve.inverse(float(shifted[2])) - 1 + 2 / steps) < 1e-6
        print(f"flux2 shift seq_len={seq_len} steps={steps}: factor={factor:.4f} ok")

    constant = TypeAdapter(Shift).validate_python(3.0)
    assert isinstance(constant, ConstantShift) and constant.shift_value == 3.0
    print("[green]shift checks passed[/green]")
