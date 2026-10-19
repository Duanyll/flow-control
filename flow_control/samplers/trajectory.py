"""Trajectory nodes: the row moves from its σ to wherever the node ends."""

from __future__ import annotations

import math
from functools import cached_property
from typing import Annotated, Any, Literal

import torch
from pydantic import Field, model_validator

from flow_control.utils.condition_image import ConditionImage, ConditionImageSpec
from flow_control.utils.registry import Registry, RegistryUnion

from .calls import Calls, gather
from .node import BaseNode, Frame, Row, randn
from .shift import ConstantShift, Shift
from .transition import BaseTransition, Euler, Transition


class BaseTrajectory(BaseNode):
    def __call__(self, row: Row) -> Calls[None]:
        with self.enter(row) as frame:
            yield from self.run(row, frame)

    def run(self, row: Row, frame: Any, /) -> Calls[None]:  # frame: self.frame_type
        raise NotImplementedError


trajectory_registry: Registry[BaseTrajectory] = Registry(
    "trajectory", base=BaseTrajectory
)
Trajectory = Annotated[
    BaseTrajectory,
    RegistryUnion(trajectory_registry, "type", list_as=("sequential", "steps")),
]


class GridFrame(Frame):
    sigmas: list[float]
    index: int
    window: tuple[int, int]  # set by SdeWindowGrid


class BaseGrid(BaseTrajectory):
    """``steps`` evaluations from the row's σ to ``sigma_end`` on a shifted grid."""

    steps: int = 50
    sigma_end: float = 0.0
    shift: Shift = ConstantShift()
    custom_sigmas: list[float] | None = None
    """Explicit grid of ``steps + 1`` sigmas from the row's σ to ``sigma_end``; skips ``shift``."""
    frame_type = GridFrame

    def sigmas(self, row: Row) -> list[float]:
        start, end = row["sigma"], self.sigma_end
        if self.custom_sigmas is not None:
            last = self.custom_sigmas[-1]
            if len(self.custom_sigmas) != self.steps + 1 or not math.isclose(
                last, end, abs_tol=1e-6
            ):
                raise ValueError(
                    f"custom_sigmas must hold steps + 1 = {self.steps + 1} entries "
                    f"ending at sigma_end {end}; got {len(self.custom_sigmas)} "
                    f"entries ending at {last}."
                )
            # Official grids label the pure noise below 1 (HiDream 0.999, Cosmos 0.995).
            row["sigma"] = self.custom_sigmas[0]
            return list(self.custom_sigmas)
        # The shift's terminal stretch is defined on the full 1 -> 0 grid.
        curve = self.shift.curve(row, num_steps=round(self.steps / abs(start - end)))
        u = torch.linspace(curve.inverse(start), curve.inverse(end), self.steps + 1)
        return [start, *curve(u[1:-1]).tolist(), end]


@trajectory_registry.register("grid")
class Grid(BaseGrid):
    type: Literal["grid"] = "grid"
    transition: Transition = Euler()

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.sigmas = self.sigmas(row)
        for grid.index in range(self.steps):
            yield from self.step(row, grid)(row, grid.sigmas[grid.index + 1])

    def step(self, row: Row, grid: GridFrame, /) -> BaseTransition:
        return self.transition

    def estimate_calls(self, row: Row) -> int:
        return self.steps * self.transition.estimate_calls(row)


def select_sde_window(
    num_steps: int,
    size: int,
    window_range: tuple[int, int] | None,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    """A random ``size``-step window inside ``window_range`` (default: all but the terminal step)."""
    start, end = window_range or (0, num_steps - 1)
    if not 0 <= start < end < num_steps or not 0 < size <= end - start:
        raise ValueError(
            f"SDE window size={size} does not fit range=({start}, {end}) of {num_steps} steps."
        )
    device = generator.device if generator is not None else "cpu"
    first = torch.randint(start, end - size + 1, (), generator=generator, device=device)
    return int(first.item()), int(first.item()) + size


@trajectory_registry.register("sde_window_grid")
class SdeWindowGrid(Grid):
    """A grid whose steps inside a random window run the transition with ``eta``."""

    type: Literal["sde_window_grid"] = "sde_window_grid"
    size: int
    window_range: tuple[int, int] | None = None
    eta: float

    @model_validator(mode="after")
    def _transition_has_eta(self) -> SdeWindowGrid:
        if "eta" not in type(self.transition).model_fields:
            raise ValueError(
                f"sde_window_grid needs a transition with `eta`; {self.transition.type!r} has none."
            )
        return self

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.window = select_sde_window(
            self.steps, self.size, self.window_range, row["_generator"]
        )
        yield from super().run(row, grid)

    def step(self, row: Row, grid: GridFrame, /) -> BaseTransition:
        start, end = grid.window
        return self._sde if start <= grid.index < end else self.transition

    @cached_property
    def _sde(self) -> BaseTransition:
        return self.transition.model_copy(update={"eta": self.eta})


@trajectory_registry.register("sequential")
class Sequential(BaseTrajectory):
    type: Literal["sequential"] = "sequential"
    steps: list[Trajectory]

    def run(self, row: Row, frame: Frame, /) -> Calls[None]:
        for child in self.steps:
            yield from child(row)


@trajectory_registry.register("renoise")
class Renoise(BaseTrajectory):
    """The forward process x_σ -> x_σend for σend >= σ; no model call."""

    type: Literal["renoise"] = "renoise"
    sigma_end: float

    def run(self, row: Row, frame: Frame, /) -> Calls[None]:
        a, b = row["sigma"], self.sigma_end
        if b != a:
            c = (1 - b) / (1 - a)
            row["noisy_latents"] = c * row["noisy_latents"] + math.sqrt(
                b**2 - (c * a) ** 2
            ) * randn(row)
            row["sigma"] = b
        yield from gather([])


@trajectory_registry.register("init_latent")
class InitLatent(BaseTrajectory):
    """x = the source image's latents at σ = 0, then renoise to ``sigma_end``."""

    type: Literal["init_latent"] = "init_latent"
    source: ConditionImageSpec = Field(alias="from")
    sigma_end: float = 1.0

    def run(self, row: Row, frame: Frame, /) -> Calls[None]:
        latents = ConditionImage.parse(self.source).latents(row).float()
        if latents.shape != row["noisy_latents"].shape:
            raise ValueError(
                f"init_latent source {self.source} has shape {tuple(latents.shape)}, "
                f"the row's latents {tuple(row['noisy_latents'].shape)}."
            )
        row["noisy_latents"], row["sigma"] = latents, 0.0
        yield from Renoise(sigma_end=self.sigma_end)(row)
