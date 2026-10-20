"""Reward placement and slots (``docs/rollout-reward-design.md``).

A reward's *placement* says where a call runs: a :class:`LocalReward` on the
device worker, batched across rows and groups; a :class:`RemoteReward` on the
reward loop, one request per item, concurrently. Its *slot* says which config
fields accept it: a :class:`BaseScorer` scores a row as ``[C]`` raw on its
labels, a :class:`BaseJudge` returns P(a ≻ b) for a pair of rows. Every
registered reward is one slot class and one placement, checked at registration.
"""

import asyncio
from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Any, Self, cast

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    PositiveInt,
    RootModel,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema
from torch import Tensor

from flow_control.samplers.node import Row
from flow_control.utils.registry import Registry, RegistryUnion
from flow_control.utils.tensor import deep_move_to_device

from .normalize import IdentityNormalize, Normalize

if TYPE_CHECKING:
    from flow_control.training.rollouts.runner import RewardContext


class LocalReward[I, O](BaseModel):
    """Runs on the device worker (main thread), batched across rows and groups.

    ``run_batch`` gets up to ``batch_size`` items. Its inputs are the rows as
    they are: their tensors may be on CPU storage or on the device, so move what
    you need to your own device. The worker moves the outputs to CPU.
    """

    batch_size: PositiveInt = 32
    """Items per ``run_batch()``; the worker fills it across groups."""

    def load(self, device: torch.device) -> None:
        """Load weights onto *device*; called once, before the first ``run_batch()``."""

    def run_batch(self, items: list[I], /) -> list[O]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_batch()."
        )


class RemoteReward[I, O](BaseModel):
    """Runs on the reward loop (background thread): one request per item, concurrently.

    The rows inside an item arrive as CPU snapshots filtered to :attr:`fields`,
    taken on the device worker.
    """

    @property
    @abstractmethod
    def fields(self) -> set[str]:
        """Row keys this reward reads."""

    def load(self, device: torch.device) -> None:
        """Prepare clients; a remote reward never touches *device*."""

    async def run_one(self, item: I, /) -> O:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_one()."
        )


type Reward[I, O] = LocalReward[I, O] | RemoteReward[I, O]


def run_local_batch[I, O](reward: LocalReward[I, O], items: list[I]) -> list[O]:
    """``reward.run_batch(items)`` under ``no_grad``: one output per item, moved
    to CPU (the runner's device worker and the reward server)."""
    with torch.no_grad():
        outputs = reward.run_batch(items)
    if len(outputs) != len(items):
        raise ValueError(
            f"{type(reward).__name__}.run_batch() returned {len(outputs)} outputs "
            f"for {len(items)} items; one per item."
        )
    return deep_move_to_device(outputs, torch.device("cpu"))


def load_rewards(config: BaseModel, device: torch.device) -> None:
    """Call ``load(device)`` on every reward found in the pydantic tree of
    *config*: nested models, ``RootModel`` roots, lists, tuples and dict values,
    a reward's own fields included (a scorer wrapping a judge, say)."""

    def walk(value: Any) -> None:
        if isinstance(value, LocalReward | RemoteReward):
            value.load(device)
        children: Iterable[Any]
        if isinstance(value, RootModel):
            children = [value.root]
        elif isinstance(value, BaseModel):
            children = [getattr(value, name) for name in type(value).model_fields]
        elif isinstance(value, dict):
            children = value.values()
        elif isinstance(value, list | tuple):
            children = value
        else:
            return
        for child in children:
            walk(child)

    walk(config)


class BaseScorer(BaseModel):
    """The scorer slot: a ``Reward[Row, Tensor]`` returning ``[C]`` raw on :attr:`labels`.

    A scorer returns raw, model-native scores; :meth:`Scorers.normalize` applies
    :attr:`normalize`, never the scorer itself.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    name: str | None = None
    """Prefix of this scorer's labels in logs and metrics; default ``type``."""
    weight: float = 1.0
    """Scales every component weight."""
    normalize: Normalize = Field(default_factory=IdentityNormalize)
    """Element-wise on the raw score; a class may default it (pickscore: affine 1/26)."""

    @property
    def label(self) -> str:
        return self.name or self.type

    @property
    def labels(self) -> list[str]:
        """Component names; a single component logs as :attr:`label`."""
        return ["score"]

    @property
    def weights(self) -> list[float]:
        """Per-component weights; 0 logs a component without training on it."""
        return [1.0] * len(self.labels)

    @property
    @abstractmethod
    def fields(self) -> set[str]:
        """Row keys this scorer reads (what a reward server snapshots for it)."""

    async def score(self, ctx: "RewardContext", rows: list[Row]) -> Tensor:
        """``[N, C]`` raw float32 on CPU."""
        # The registries admit only classes with exactly one placement.
        reward = cast("Reward[Row, Tensor]", self)
        raw = torch.stack(await ctx.run(reward, rows)).float().cpu()
        if raw.shape != (len(rows), len(self.labels)):
            raise ValueError(
                f"Scorer {self.label!r} returned {list(raw.shape[1:])} per row; "
                f"expected [{len(self.labels)}], one value per label {self.labels}."
            )
        return raw


class BaseJudge(BaseModel):
    """The judge slot: a ``Reward[tuple[Row, Row], float]`` returning P(a ≻ b), 0.5 a tie."""

    model_config = ConfigDict(extra="forbid")

    type: str

    @property
    @abstractmethod
    def fields(self) -> set[str]:
        """Row keys this judge reads from either row of a pair."""


def _one_placement(cls: type) -> None:
    placements = [p.__name__ for p in (LocalReward, RemoteReward) if issubclass(cls, p)]
    if len(placements) != 1:
        raise TypeError(
            f"{cls.__name__} must subclass exactly one of LocalReward (device worker, "
            "batched) and RemoteReward (reward loop, one request per item); it "
            f"subclasses {' and '.join(placements) or 'neither'}."
        )


# Members register at their definition sites via ``@scorer_registry.register(tag)``.
scorer_registry: Registry[BaseScorer] = Registry(
    "scorer", base=BaseScorer, check=_one_placement
)
judge_registry: Registry[BaseJudge] = Registry(
    "judge", base=BaseJudge, check=_one_placement
)

Scorer = Annotated[BaseScorer, RegistryUnion(scorer_registry, "type")]
Judge = Annotated[BaseJudge, RegistryUnion(judge_registry, "type")]


class Scorers(RootModel):
    """Scorers run side by side on every row; a single scorer config is shorthand
    for a one-item list. Components concatenate in list order."""

    # Built at first use: the scorers register after this module is imported.
    model_config = ConfigDict(defer_build=True)

    root: list[Scorer]

    @model_validator(mode="before")
    @classmethod
    def _single_scorer(cls, value: Any) -> Any:
        return [value] if isinstance(value, dict | str | BaseScorer) else value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Advertise the single-scorer shorthand next to the list.
        schema = handler(core_schema)
        target = handler.resolve_ref_schema(schema)
        array = dict(target)
        target.clear()
        target.update(
            title=array.get("title", cls.__name__), anyOf=[array["items"], array]
        )
        return schema

    @model_validator(mode="after")
    def _unique_labels(self) -> Self:
        seen: set[str] = set()
        for scorer in self.root:
            if scorer.label in seen:
                raise ValueError(
                    f"Two scorers are labelled {scorer.label!r}; give one a distinct "
                    "`name`."
                )
            seen.add(scorer.label)
        return self

    def __bool__(self) -> bool:
        return bool(self.root)

    @property
    def labels(self) -> list[str]:
        """``<label>`` for a one-component scorer, ``<label>/<component>`` otherwise."""
        return [
            scorer.label if len(scorer.labels) == 1 else f"{scorer.label}/{component}"
            for scorer in self.root
            for component in scorer.labels
        ]

    @property
    def weights(self) -> list[float]:
        return [scorer.weight * w for scorer in self.root for w in scorer.weights]

    async def score(self, ctx: "RewardContext", rows: list[Row]) -> Tensor:
        """``[N, C]`` raw float32 on CPU, ``C = len(self.labels)``."""
        if not self.root:
            return torch.zeros(len(rows), 0)
        parts = await asyncio.gather(*(s.score(ctx, rows) for s in self.root))
        return torch.cat(parts, dim=-1)

    def normalize(self, raw: Tensor) -> Tensor:
        """Each scorer's :attr:`~BaseScorer.normalize` on its own columns of ``raw``."""
        if not self.root:
            return raw
        blocks = raw.split([len(s.labels) for s in self.root], dim=-1)
        return torch.cat(
            [s.normalize.apply(b) for s, b in zip(self.root, blocks, strict=True)],
            dim=-1,
        )


if __name__ == "__main__":
    from typing import Literal

    from rich import print

    from .normalize import AffineNormalize

    @scorer_registry.register("_test_local")
    class _Local(BaseScorer, LocalReward[Row, Tensor]):
        """Two components ``[v, -v]``; doubles by default."""

        type: Literal["_test_local"] = "_test_local"
        normalize: Normalize = Field(default_factory=lambda: AffineNormalize(scale=2))

        @property
        def labels(self) -> list[str]:
            return ["value", "negated"]

        @property
        def weights(self) -> list[float]:
            return [1.0, 0.5]

        @property
        def fields(self) -> set[str]:
            return {"value"}

        def run_batch(self, rows: list[Row]) -> list[Tensor]:
            return [torch.tensor([row["value"], -row["value"]]) for row in rows]

    @scorer_registry.register("_test_remote")
    class _Remote(BaseScorer, RemoteReward[Row, Tensor]):
        type: Literal["_test_remote"] = "_test_remote"

        @property
        def fields(self) -> set[str]:
            return {"value"}

        async def run_one(self, row: Row) -> Tensor:
            await asyncio.sleep(0.01 * (3 - row["value"]))  # finishes out of order
            return torch.tensor([10.0 * row["value"]], dtype=torch.float64)

    class _Ctx:
        def __init__(self) -> None:
            self.batches: list[int] = []

        async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]:
            if isinstance(reward, LocalReward):
                self.batches.append(len(items))
                return reward.run_batch(items)
            return list(await asyncio.gather(*(reward.run_one(i) for i in items)))

    # A single scorer (a dict or a bare tag) is a one-item list.
    for single in ({"type": "_test_local"}, "_test_local", _Local()):
        parsed = Scorers.model_validate(single)
        assert len(parsed.root) == 1 and isinstance(parsed.root[0], _Local), parsed
    variants = Scorers.model_json_schema()["anyOf"]  # one scorer, or a list
    assert len(variants) == 2 and variants[1]["type"] == "array", variants

    scorers = Scorers.model_validate(
        [
            {"type": "_test_local", "name": "loc", "weight": 0.5},
            {"type": "_test_remote", "weight": 2.0},
        ]
    )
    assert scorers.labels == ["loc/value", "loc/negated", "_test_remote"]
    assert scorers.weights == [0.5, 0.25, 2.0], scorers.weights

    # score: [N, C] float32 on CPU, rows in order; normalize per column block.
    stub = _Ctx()
    ctx = cast("RewardContext", stub)
    rows: list[Row] = [{"value": float(v)} for v in range(3)]
    raw = asyncio.run(scorers.score(ctx, rows))
    assert raw.dtype == torch.float32 and raw.device.type == "cpu"
    expected = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 10.0], [2.0, -2.0, 20.0]])
    assert torch.equal(raw, expected), raw
    assert stub.batches == [3], stub.batches
    assert torch.equal(scorers.normalize(raw), expected * torch.tensor([2, 2, 1]))
    custom = Scorers.model_validate(
        [{"type": "_test_remote", "normalize": {"type": "affine", "offset": 1.0}}]
    )
    assert torch.equal(custom.normalize(raw[:, 2:]), raw[:, 2:] + 1)

    class _Flat(_Local):
        def run_batch(self, rows: list[Row]) -> list[Tensor]:
            return [torch.zeros(1, 2) for _ in rows]  # the old [1, C] convention

    try:
        asyncio.run(_Flat(name="flat").score(ctx, rows))
    except ValueError as exc:
        assert "'flat' returned [1, 2]" in str(exc), exc
    else:
        raise AssertionError("a [1, C] score must be rejected")
    empty = asyncio.run(Scorers([]).score(ctx, rows))
    assert empty.shape == (3, 0) and Scorers([]).normalize(empty).shape == (3, 0)

    # load_rewards reaches rewards through nested models, lists and dicts.
    loaded: list[str] = []

    class _Loading(_Local):
        def load(self, device: torch.device) -> None:
            loaded.append(f"{self.name}@{device.type}")

    class _Config(BaseModel):
        scorers: Scorers
        nested: dict[str, list[_Loading]]

    config = _Config(
        scorers=Scorers([_Loading(name="listed")]),
        nested={"a": [_Loading(name="dicted")]},
    )
    load_rewards(config, torch.device("cpu"))
    assert loaded == ["listed@cpu", "dicted@cpu"], loaded

    print("[green]rewards.base self-test passed[/green]")
