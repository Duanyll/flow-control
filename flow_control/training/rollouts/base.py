"""Rollout programs (``docs/rollout-reward-design.md``).

A rollout program (``rollout.type``) turns one prompt into its K scored
rollouts. :meth:`BaseRollout.sample` makes every model call of the group and
returns whatever :meth:`BaseRollout.score` needs; ``score`` calls rewards only
and returns the rows the trainer trains on, their ``reward_raw`` / ``reward``
(``[C]`` on :attr:`BaseRollout.labels`, CPU) written by :func:`write`.
"""

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from flow_control.rewards.base import Scorers
from flow_control.samplers.node import Row
from flow_control.samplers.trajectory import Trajectory
from flow_control.utils.registry import Registry, RegistryUnion

if TYPE_CHECKING:
    from flow_control.training.rollouts.runner import RewardContext, SampleContext


class BaseRollout[G](BaseModel):
    """A rollout program; ``G`` is what :meth:`sample` hands to :meth:`score`."""

    # Built at first use, so the schema sees the judges and scorers that plugins
    # register after this module is imported.
    model_config = ConfigDict(extra="forbid", defer_build=True)

    type: str
    sampler: Trajectory
    """The policy sampler: ``ctx.rollouts()`` samples with it."""
    reward: Scorers = Field(default_factory=lambda: Scorers([]))
    """Scorers on every policy rollout."""

    @property
    def labels(self) -> list[str]:
        """The reward components the advantage and the logs see."""
        return self.reward.labels

    @property
    def weights(self) -> list[float]:
        """Per-component weights of the advantage; 0 logs a component only."""
        return self.reward.weights

    async def sample(self, ctx: "SampleContext", /) -> G:
        """Every model call of the group."""
        raise NotImplementedError(f"{type(self).__name__} does not implement sample().")

    async def score(self, ctx: "RewardContext", group: G, /) -> list[Row]:
        """Rewards only: the rows to train on, their rewards set by :func:`write`."""
        raise NotImplementedError(f"{type(self).__name__} does not implement score().")


# Members register at their definition sites via ``@rollout_registry.register(tag)``.
rollout_registry: Registry[BaseRollout] = Registry("rollout", base=BaseRollout)

Rollout = Annotated[BaseRollout, RegistryUnion(rollout_registry, "type")]


def write(rows: list[Row], raw: Tensor, normalized: Tensor) -> None:
    """Set each row's ``reward_raw`` / ``reward`` to its ``[C]`` row of the ``[N, C]``
    *raw* / *normalized*, float on CPU."""
    raw, normalized = raw.detach().float().cpu(), normalized.detach().float().cpu()
    for row, r, n in zip(rows, raw, normalized, strict=True):
        row["reward_raw"], row["reward"] = r, n


@rollout_registry.register("pointwise")
class PointwiseRollout(BaseRollout[list[Row]]):
    """Every rollout scored on its own by :attr:`reward`."""

    type: Literal["pointwise"] = "pointwise"

    @model_validator(mode="after")
    def _has_scorers(self) -> Self:
        # Subclasses (round_robin) bring components of their own.
        if type(self) is PointwiseRollout and not self.reward:
            raise ValueError("`pointwise` needs at least one scorer in `reward`.")
        return self

    async def sample(self, ctx: "SampleContext") -> list[Row]:
        return await ctx.rollouts()

    async def score(self, ctx: "RewardContext", rows: list[Row]) -> list[Row]:
        raw = await self.reward.score(ctx, rows)
        write(rows, raw, self.reward.normalize(raw))
        return rows
