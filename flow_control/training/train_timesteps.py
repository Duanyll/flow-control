"""Training-timestep selection for endpoint-based rollout trainers.

A :class:`TrainTimesteps` decides, per rollout, at which timesteps the clean
endpoint is re-noised and trained on. It is the second axis of the trainer
layering next to the :class:`~flow_control.training.endpoint.Objective`: a
*grid* draw
returns rollout-grid indices (the trainer then evaluates the point through the
rollout plan's guided step), a *continuous* draw returns free timesteps (the
trainer evaluates them through the train predictor directly).

Conventions (flow_control): ``sigmas`` is the executed rollout grid in plan
order, ``sigmas[0]`` the noisiest (``rollout.sampling_plan[i].sigma``), and a
timestep ``t in [0, 1]`` has ``1`` = pure noise. All draws use CPU tensors.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Literal, NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from flow_control.utils.registry import Registry, RegistryUnion

from .weighting import LogitNormalTimestepWeighting, TimestepWeighting


class TrainTimestep(NamedTuple):
    sigma: float
    """Training timestep, ``1`` = pure noise."""
    grid_index: int | None
    """Index of ``sigma`` on the rollout grid, or ``None`` for a continuous draw."""


class BaseTrainTimesteps(BaseModel, ABC):
    type: str
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def draw(self, sigmas: list[float]) -> list[TrainTimestep]:
        """Training timesteps for one rollout whose executed grid is ``sigmas``.

        ``sigmas`` is in plan order (index 0 = noisiest). Called once per
        rollout per inner epoch; the trainer builds one train item per entry.
        """
        ...


train_timesteps_registry: Registry[BaseTrainTimesteps] = Registry(
    "train_timesteps", base=BaseTrainTimesteps
)


@train_timesteps_registry.register("grid")
class GridTimesteps(BaseTrainTimesteps):
    """Draw training timesteps from the rollout's own sigma grid.

    The eligible window is the index range ``[lo, hi)`` with ``lo = 1`` if
    :attr:`exclude_first` else ``0`` and ``hi = max(lo + 1, int(len(sigmas) *
    window))``, i.e. the noisiest ``window`` fraction of the grid (AWM's
    ``timestep_fraction``; ``discrete_wo_init`` is ``exclude_first``). ``k``
    indices are drawn from it: ``count`` if set (clamped to the window size),
    else ``max(1, int(n * fraction))``. ``random`` draws a uniform distinct
    subset; ``stratified`` splits the window into ``k`` equal bins and draws one
    index uniformly per bin for even noise coverage (Flow-Factory's discrete
    stratified sampling).

    Distributed invariant: with index-based windows the number of items per
    rollout depends only on ``len(sigmas)``, so all ranks build the same item
    count for equal step counts and the per-microbatch model collectives line
    up. This replaces NFT's sigma-threshold window (``timestep_range``), whose
    eligible count depended on the rollout's sigma values and hence, through a
    resolution-dependent shift, on the image size. On a uniform grid the two
    agree; on a shifted grid they differ by an index or two.
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

    def draw(self, sigmas: list[float]) -> list[TrainTimestep]:
        lo = 1 if self.exclude_first else 0
        hi = max(lo + 1, int(len(sigmas) * self.window))
        if hi > len(sigmas):
            raise RuntimeError(
                f"Rollout grid with {len(sigmas)} step(s) has no training "
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
        return [TrainTimestep(sigmas[i], i) for i in indices]


@train_timesteps_registry.register("continuous")
class ContinuousTimesteps(BaseTrainTimesteps):
    """Draw ``count`` timesteps from a :class:`TimestepWeighting`, ignoring the
    grid (RAM's power-law sampler; AWM's ``weighting`` mode). The trainer
    evaluates these through the train predictor, not the rollout plan."""

    type: Literal["continuous"] = "continuous"
    count: PositiveInt
    """Timesteps per rollout."""
    weighting: TimestepWeighting = LogitNormalTimestepWeighting()

    def draw(self, sigmas: list[float]) -> list[TrainTimestep]:
        return [
            TrainTimestep(float(t), None)
            for t in self.weighting.sample_timesteps(self.count)
        ]


TrainTimesteps = Annotated[
    BaseTrainTimesteps, RegistryUnion(train_timesteps_registry, "type")
]


if __name__ == "__main__":
    from pydantic import TypeAdapter, ValidationError
    from rich import print

    torch.manual_seed(0)
    adapter = TypeAdapter(TrainTimesteps)
    uniform = [1.0 - i / 10 for i in range(10)]
    shifted = [s / (s + (1 - s) / 3) for s in uniform]  # same length, other values

    # Window bounds and exclude_first, both modes, several window widths.
    for mode in ("random", "stratified"):
        for window in (1.0, 0.9, 0.5, 0.05):
            for exclude_first in (False, True):
                grid = GridTimesteps(
                    mode=mode, window=window, exclude_first=exclude_first
                )
                lo = 1 if exclude_first else 0
                hi = max(lo + 1, int(len(uniform) * window))
                for _ in range(20):
                    items = grid.draw(uniform)
                    assert len(items) == hi - lo, (mode, window, exclude_first, items)
                    for sigma, index in items:
                        assert index is not None and lo <= index < hi, items
                        assert sigma == uniform[index]
                    assert len({i for _, i in items}) == len(items), items
                # Item count depends only on len(sigmas), never on the values.
                assert len(grid.draw(shifted)) == len(grid.draw(uniform))

    # Stratified with k == n covers every bin exactly once.
    for exclude_first in (False, True):
        strat = GridTimesteps(mode="stratified", exclude_first=exclude_first)
        lo = 1 if exclude_first else 0
        for _ in range(20):
            drawn = sorted(i for _, i in strat.draw(uniform) if i is not None)
            assert drawn == list(range(lo, 10)), drawn
    # Stratified with k < n: one index per equal bin. linspace(0, 10, 5) gives
    # boundaries 0, 2.5, 5, 7.5, 10 -> integer bins [0,2) [2,5) [5,7) [7,10).
    strat = GridTimesteps(mode="stratified", count=4)
    bins = ({0, 1}, {2, 3, 4}, {5, 6}, {7, 8, 9})
    for _ in range(50):
        idx = sorted(i for _, i in strat.draw(uniform) if i is not None)
        assert all(i in b for i, b in zip(idx, bins, strict=True)), idx

    # count clamps to the window size; fraction derives the count.
    assert len(GridTimesteps(count=100).draw(uniform)) == 10
    assert len(GridTimesteps(count=100, window=0.5).draw(uniform)) == 5
    assert len(GridTimesteps(count=3).draw(uniform)) == 3
    assert len(GridTimesteps(fraction=0.25).draw(uniform)) == 2
    assert len(GridTimesteps(fraction=0.01).draw(uniform)) == 1
    assert len(GridTimesteps(window=0.01).draw(uniform)) == 1
    assert (
        GridTimesteps(window=0.01, exclude_first=True).draw(uniform)[0].grid_index == 1
    )

    # Too few steps for the window is an error, not a silent out-of-range index.
    for grid, sigmas in (
        (GridTimesteps(), []),
        (GridTimesteps(exclude_first=True), [1.0]),
    ):
        try:
            grid.draw(sigmas)
        except RuntimeError as exc:
            print(f"[green]short grid raises:[/] {exc}")
        else:
            raise AssertionError(f"{grid}.draw({sigmas}) must raise")

    # Config validation: ranges, positive counts, no legacy flat keys.
    for bad in (
        {"type": "grid", "fraction": 0.0},
        {"type": "grid", "window": 1.5},
        {"type": "grid", "count": 0},
        {"type": "continuous", "count": 0},
        {"type": "grid", "timestep_range": 0.5},
    ):
        try:
            adapter.validate_python(bad)
        except ValidationError as exc:
            print(f"[green]{bad} rejected:[/] {exc.errors()[0]['msg']}")
        else:
            raise AssertionError(f"{bad} must not validate")

    # Continuous draws ignore the grid and carry no index.
    cont = adapter.validate_python({"type": "continuous", "count": 7})
    assert isinstance(cont, ContinuousTimesteps)
    items = cont.draw(uniform)
    assert len(items) == 7 and all(i is None for _, i in items)
    assert all(0.0 <= s <= 1.0 for s, _ in items)
    assert len(cont.draw([])) == 7
    power = adapter.validate_python(
        {"type": "continuous", "count": 3, "weighting": {"type": "power_law"}}
    )
    assert len(power.draw(uniform)) == 3

    # Registry shorthand and defaults.
    assert isinstance(adapter.validate_python("grid"), GridTimesteps)
    assert adapter.validate_python("grid") == GridTimesteps()

    print(
        GridTimesteps(mode="stratified", window=0.9, exclude_first=True).draw(uniform)
    )
    print(cont.draw(uniform))
    print("[green]train_timesteps checks passed[/green]")
