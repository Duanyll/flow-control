"""Rows, call-chain frames and the node base every sampler layer shares."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict

type Row = dict[str, Any]
type Observer = Callable[[Row, Any], None]


def derived[T](row: Row, key: str, make: Callable[[], T]) -> T:
    """A branch value built once and kept on the row."""
    if key not in row:
        row[key] = make()
    return row[key]


def follow(sub: Row, row: Row) -> Row:
    """Move a derived row to the parent's point and call chain."""
    sub.update(
        noisy_latents=row["noisy_latents"], sigma=row["sigma"], _frame=row["_frame"]
    )
    sub.pop("sigma_next", None)
    if "sigma_next" in row:  # inside a step: the branch shares the flow-map target
        sub["sigma_next"] = row["sigma_next"]
    return sub


def data(row: Row) -> Row:
    """The row without runtime keys: the start of a new point of the same sample."""
    return {key: value for key, value in row.items() if not key.startswith("_")}


def randn(row: Row) -> torch.Tensor:
    """Noise shaped like the row's latents, drawn from the row's generator."""
    x = row["noisy_latents"]
    return torch.randn(
        x.shape, dtype=x.dtype, device=x.device, generator=row["_generator"]
    )


class Frame:
    """One active node on a row's call chain."""

    root: RootFrame

    def __init__(self, node: object, parent: Frame | None) -> None:
        self.node, self.parent = node, parent
        if parent is not None:
            self.root = parent.root

    def snapshot(self) -> Frame:
        """A copy of the chain with every tensor dropped; the root is shared."""
        if self.parent is None:
            return self
        clone = copy.copy(self)
        clone.parent = self.parent.snapshot()
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(clone, key, None)
        return clone

    def rerooted(self, root: RootFrame) -> Frame:
        """A copy of the chain under *root*: a recorded chain replayed by another
        engine, whose leaves then run under that root's variant (None: the
        active one) rather than the variant that sampled the row."""
        if self.parent is None:
            return root
        clone = copy.copy(self)
        clone.parent = self.parent.rerooted(root)
        clone.root = root
        return clone


class RootFrame(Frame):
    """The chain root the engine places on a row."""

    def __init__(
        self,
        engine: object,
        observers: Sequence[tuple[type, Observer]] = (),
        on_call: Callable[[Row], None] | None = None,
        variant: str | None = None,
    ) -> None:
        super().__init__(engine, None)
        self.root = self
        self.observers, self.on_call = list(observers), on_call
        self.variant = variant
        """The variant this row's ``Model(variant=None)`` leaves run under; None is the adapter's active one."""


def _frames(frame: Frame | None) -> Iterator[Frame]:
    while frame is not None:
        yield frame
        frame = frame.parent


def caller(row: Row, kind: type | tuple[type, ...], who: str) -> Frame:
    """The nearest enclosing frame whose node is a `kind`; otherwise raise naming `who`."""
    for frame in _frames(row["_frame"]):
        if isinstance(frame.node, kind):
            return frame
    chain = " < ".join(type(frame.node).__name__ for frame in _frames(row["_frame"]))
    raise ValueError(f"{who}, but the call chain is {chain}.")


class BaseNode(BaseModel):
    """A sampler node: a config whose call pushes a frame on the row."""

    model_config = ConfigDict(
        extra="forbid",
        defer_build=True,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )
    type: str
    frame_type: ClassVar[type[Frame]] = Frame

    @contextmanager
    def enter(self, row: Row) -> Iterator[Any]:
        frame = row["_frame"] = self.frame_type(self, row["_frame"])
        try:
            yield frame
        finally:
            row["_frame"] = frame.parent
        for kind, observer in frame.root.observers:  # only on normal return
            if isinstance(self, kind):
                observer(row, frame)

    def estimate_calls(self, row: Row) -> int:
        """Model calls this node will make for the row; an estimate for progress."""
        return sum(
            child.estimate_calls(row)
            for child in children(self)
            if isinstance(child, BaseNode)
        )


def _models(value: object) -> Iterator[BaseModel]:
    if isinstance(value, BaseModel):
        yield value
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _models(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _models(item)


def children(node: BaseModel) -> Iterator[BaseModel]:
    """The direct child models of a node: fields, lists and dicts."""
    for name in type(node).model_fields:
        yield from _models(getattr(node, name))


def walk(tree: BaseModel) -> Iterator[BaseModel]:
    """Every model in the config tree, depth first."""
    yield tree
    for child in children(tree):
        yield from walk(child)
