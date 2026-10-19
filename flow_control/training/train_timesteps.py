"""Training-timestep selection for endpoint-based rollout trainers.

A :class:`TrainTimesteps` decides, per rollout, at which sigmas the clean
endpoint is re-noised and trained on: a *grid* draw picks step starts of the
executed rollout grid, a *continuous* draw samples free sigmas. ``1`` = pure
noise; all draws use CPU tensors.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from flow_control.utils.registry import Registry, RegistryUnion

from .weighting import LogitNormalTimestepWeighting, TimestepWeighting


class BaseTrainTimesteps(BaseModel, ABC):
    type: str
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def draw(self, sigmas: list[float]) -> list[float]:
        """Training sigmas for one rollout whose executed grid is ``sigmas`` (terminal included)."""
        ...


train_timesteps_registry: Registry[BaseTrainTimesteps] = Registry(
    "train_timesteps", base=BaseTrainTimesteps
)


@train_timesteps_registry.register("grid")
class GridTimesteps(BaseTrainTimesteps):
    """Draw training sigmas from the rollout's own grid.

    The eligible window is the step-start index range ``[lo, hi)`` with
    ``lo = 1`` if :attr:`exclude_first` else ``0`` and ``hi = max(lo + 1,
    int(steps * window))``, i.e. the noisiest ``window`` fraction of the grid
    (AWM's ``timestep_fraction``; ``discrete_wo_init`` is ``exclude_first``).
    ``k`` indices are drawn from it: ``count`` if set (clamped to the window
    size), else ``max(1, int(n * fraction))``. ``random`` draws a uniform
    distinct subset; ``stratified`` splits the window into ``k`` equal bins and
    draws one index uniformly per bin (Flow-Factory's discrete stratified
    sampling).

    Distributed invariant: the number of items per rollout depends only on the
    step count, so all ranks build the same item count for equal step counts
    and the per-microbatch model collectives line up.
    """

    type: Literal["grid"] = "grid"
    count: PositiveInt | None = None
    """Timesteps per rollout. ``None`` derives it from the window and ``fraction``."""
    fraction: float = 1.0
    """Fraction of the window to train on when ``count`` is ``None``."""
    window: float = 1.0
    """Keep only the noisiest ``window`` fraction of the grid; ``1.0`` keeps all."""
    exclude_first: bool = False
    """Skip index 0 (the pure-noise step)."""
    mode: Literal["random", "stratified"] = "random"

    @model_validator(mode="after")
    def check_ranges(self):
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}.")
        if not 0.0 < self.window <= 1.0:
            raise ValueError(f"window must be in (0, 1], got {self.window}.")
        return self

    def draw(self, sigmas: list[float]) -> list[float]:
        starts = sigmas[:-1]
        lo = 1 if self.exclude_first else 0
        hi = max(lo + 1, int(len(starts) * self.window))
        if hi > len(starts):
            raise RuntimeError(
                f"Rollout grid with {len(starts)} step(s) has no training "
                f"timestep in window [{lo}, {hi}); need at least {hi} steps."
            )
        n = hi - lo
        k = (
            min(self.count, n)
            if self.count is not None
            else max(1, int(n * self.fraction))
        )
        if self.mode == "stratified":
            boundaries = torch.linspace(lo, hi, k + 1)
            lower = boundaries[:-1].long()
            upper = boundaries[1:].long()
            offsets = (torch.rand(k) * (upper - lower)).long()
            indices = (lower + offsets).clamp(lo, hi - 1).tolist()
        else:
            indices = (lo + torch.randperm(n)[:k]).tolist()
        return [starts[i] for i in indices]


@train_timesteps_registry.register("continuous")
class ContinuousTimesteps(BaseTrainTimesteps):
    """Draw ``count`` sigmas from a :class:`TimestepWeighting`, ignoring the grid."""

    type: Literal["continuous"] = "continuous"
    count: PositiveInt
    """Timesteps per rollout."""
    weighting: TimestepWeighting = LogitNormalTimestepWeighting()

    def draw(self, sigmas: list[float]) -> list[float]:
        return self.weighting.sample_timesteps(self.count).tolist()


TrainTimesteps = Annotated[
    BaseTrainTimesteps, RegistryUnion(train_timesteps_registry, "type")
]


if __name__ == "__main__":
    from pydantic import TypeAdapter, ValidationError
    from rich import print

    torch.manual_seed(0)
    adapter = TypeAdapter(TrainTimesteps)
    uniform = [1.0 - i / 10 for i in range(11)]  # 10 steps plus the terminal 0
    shifted = [s / (s + (1 - s) / 3) for s in uniform]

    for mode in ("random", "stratified"):
        for window in (1.0, 0.9, 0.5, 0.05):
            for exclude_first in (False, True):
                grid = GridTimesteps(
                    mode=mode, window=window, exclude_first=exclude_first
                )
                lo = 1 if exclude_first else 0
                hi = max(lo + 1, int(10 * window))
                for _ in range(20):
                    drawn = grid.draw(uniform)
                    assert len(drawn) == hi - lo == len(set(drawn)), drawn
                    assert all(lo <= uniform.index(s) < hi for s in drawn), drawn
                assert len(grid.draw(shifted)) == len(grid.draw(uniform))

    # Stratified with k < n: one index per equal bin of [0, 10).
    strat = GridTimesteps(mode="stratified", count=4)
    bins = ({0, 1}, {2, 3, 4}, {5, 6}, {7, 8, 9})
    for _ in range(50):
        idx = sorted(uniform.index(s) for s in strat.draw(uniform))
        assert all(i in b for i, b in zip(idx, bins, strict=True)), idx

    assert len(GridTimesteps(count=100).draw(uniform)) == 10
    assert len(GridTimesteps(count=100, window=0.5).draw(uniform)) == 5
    assert len(GridTimesteps(fraction=0.25).draw(uniform)) == 2
    assert GridTimesteps(window=0.01, exclude_first=True).draw(uniform) == [0.9]
    for grid, sigmas in (
        (GridTimesteps(), [0.0]),
        (GridTimesteps(exclude_first=True), [1.0, 0.0]),
    ):
        try:
            grid.draw(sigmas)
        except RuntimeError as exc:
            print(f"[green]short grid raises:[/] {exc}")
        else:
            raise AssertionError(f"{grid}.draw({sigmas}) must raise")

    for bad in (
        {"type": "grid", "fraction": 0.0},
        {"type": "grid", "window": 1.5},
        {"type": "grid", "count": 0},
        {"type": "continuous", "count": 0},
    ):
        try:
            adapter.validate_python(bad)
        except ValidationError as exc:
            print(f"[green]{bad} rejected:[/] {exc.errors()[0]['msg']}")
        else:
            raise AssertionError(f"{bad} must not validate")

    cont = adapter.validate_python({"type": "continuous", "count": 7})
    assert isinstance(cont, ContinuousTimesteps)
    drawn = cont.draw(uniform)
    assert len(drawn) == 7 and all(0.0 <= s <= 1.0 for s in drawn)
    assert len(cont.draw([])) == 7
    assert adapter.validate_python("grid") == GridTimesteps()
    print(
        GridTimesteps(mode="stratified", window=0.9, exclude_first=True).draw(uniform)
    )
    print("[green]train_timesteps checks passed[/green]")
