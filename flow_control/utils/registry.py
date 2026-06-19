"""Open, plugin-extensible discriminated unions for Pydantic configs.

This is the single reusable mechanism behind every extension point in
``flow_control`` (model adapters, rewards, shifts, solvers, processor tasks,
encoders, VAEs, weightings, ...). It lets out-of-tree experimental components
register new members into a union **without editing the core union literal**,
while keeping full JSON-schema generation and satisfying pyright.

Usage at a family's definition site::

    from flow_control.utils.registry import Registry, RegistryUnion

    reward_registry: Registry[BaseReward] = Registry("reward", base=BaseReward)

    @reward_registry.register("clip_score")
    class CLIPScoreReward(BaseReward): ...

    # Statically ``Annotated[BaseReward, <metadata>]`` is just ``BaseReward`` to
    # pyright -- a plain type, no union-of-N in annotation position, no
    # ``if TYPE_CHECKING`` guards. The open union is assembled at runtime inside
    # ``RegistryUnion.__get_pydantic_core_schema__``.
    Reward = Annotated[BaseReward, RegistryUnion(reward_registry, "type")]

Out-of-tree plugins register the same way by importing the registry and applying
``@reward_registry.register(...)``; ``load_plugins`` imports those modules.

Runtime vs schema: runtime validation reads the registry *lazily at validation
time* (a member registered after the config class was built still validates, no
``model_rebuild``). The JSON schema is the union materialized at schema-build
time, so to include a plugin in a generated schema its module must be imported
*before* the schema is built (what the ``schema`` command + entrypoints do).
"""

from __future__ import annotations

import importlib
import operator
from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from typing import Annotated, Any

from pydantic import BaseModel, Discriminator, GetCoreSchemaHandler, Tag
from pydantic_core import CoreSchema, core_schema

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class Registry[T]:
    """A mutable ``tag -> class`` registry that a decorator populates.

    Core members register at import of their defining module; experimental
    plugins register at import of their (out-of-tree) module.
    """

    def __init__(self, name: str, *, base: type[T] | None = None) -> None:
        self.name = name
        # The static base used only for an optional runtime ``issubclass`` guard.
        # Pass ``base=None`` for duck-typed / Protocol families (datasets, sinks,
        # serving task templates) where a nominal subclass check is wrong.
        self._base = base
        # Widened to ``type[Any]`` so the identity-preserving ``register`` below
        # can stash an arbitrary concrete ``type[C]`` without a cast.
        self._members: dict[str, type[Any]] = {}

    def register[C](self, tag: str) -> Callable[[type[C]], type[C]]:
        # The *unbounded* method-level type var ``C`` is what keeps the decorator
        # *identity-preserving*: ``@reg.register("x")`` applied to ``class Foo(Base)``
        # leaves the name ``Foo`` typed as ``type[Foo]``, not ``type[Base]``.
        # Without it, pyright cannot narrow ``isinstance(x, Foo)`` to ``Foo`` and
        # every subclass-only field reads as "unknown attribute". ``C`` cannot be
        # bounded by ``T`` (pyright rejects a generic typevar as a constraint), so
        # we runtime-guard against ``self._base`` instead.
        def decorator(cls: type[C]) -> type[C]:
            # Keep an un-narrowed reference: the ``issubclass`` guard below
            # narrows ``cls``, which would otherwise pollute the return type.
            registered = cls
            if self._base is not None and not issubclass(cls, self._base):
                raise TypeError(
                    f"{cls.__name__} is not a subclass of {self._base.__name__} "
                    f"(registry {self.name!r}, tag {tag!r})."
                )
            existing = self._members.get(tag)
            if existing is not None and existing is not cls:
                # Running a member module via ``python -m`` imports it twice (as
                # ``__main__`` and under its canonical name), yielding two distinct
                # class objects with the same qualified name. Treat that as an
                # idempotent re-registration -- keep the first (canonical) class.
                # A genuinely different class fighting for the tag still errors.
                if existing.__qualname__ != cls.__qualname__:
                    raise ValueError(
                        f"Tag {tag!r} already registered in registry {self.name!r} "
                        f"by {existing.__name__}; cannot re-register {cls.__name__}."
                    )
                logger.debug(
                    "Ignoring duplicate registration of %r (%s) in registry %r "
                    "(module double-load).",
                    tag,
                    cls.__qualname__,
                    self.name,
                )
                return registered
            if existing is None:
                logger.debug(
                    "Registered %r -> %s in registry %r", tag, cls.__name__, self.name
                )
            self._members[tag] = registered
            return registered

        return decorator

    def members(self) -> Mapping[str, type[T]]:
        return dict(self._members)

    def get(self, tag: str) -> type[T] | None:
        return self._members.get(tag)

    def __contains__(self, tag: str) -> bool:
        return tag in self._members

    def __len__(self) -> int:
        return len(self._members)


class RegistryUnion:
    """``Annotated`` metadata that turns a base type into an *open* tagged union.

    Implements ``__get_pydantic_core_schema__``: at schema-build time it reads the
    registry, assembles ``Annotated[A | B | ..., Discriminator(disc)]`` dynamically,
    and hands it to Pydantic's own schema generator -- so JSON schema (oneOf +
    discriminator + shared ``$defs``) comes for free. Runtime validation is wrapped
    to dispatch through the registry *lazily*, so a member registered after this
    schema was built still validates with no ``model_rebuild``.
    """

    def __init__(
        self,
        registry: Registry[Any],
        discriminator: str | Callable[[Any], str],
        *,
        parser: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._registry = registry
        self._discriminator = discriminator
        # Optional custom runtime parser (e.g. the processor task x preset mixin,
        # or the dict-returning ProcessorConfig variant). When None, a default
        # registry dispatch is used. Either way runtime validation reads the
        # registry at validation time, never the union baked into the core schema.
        self._parser = parser

    def _build_annotated_union(self) -> Any:
        members = self._registry.members()
        if not members:
            raise ValueError(
                f"Registry {self._registry.name!r} is empty; import a member "
                "module (or a plugin) before building this union's schema."
            )
        tagged = [Annotated[cls, Tag(tag)] for tag, cls in members.items()]
        union: Any = tagged[0] if len(tagged) == 1 else reduce(operator.or_, tagged)
        return Annotated[union, Discriminator(self._discriminator)]

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # The union (built from the registry NOW) feeds JSON schema only --
        # generated through ``handler`` so all ``$defs`` land in the parent
        # context (shared, deduplicated, recursion-safe).
        union_schema = handler.generate_schema(self._build_annotated_union())
        # ...but runtime validation dispatches through the registry lazily.
        return core_schema.with_info_wrap_validator_function(
            self._validate, union_schema
        )

    def _validate(self, value: Any, handler: Any, info: Any) -> Any:
        if isinstance(value, dict):
            return (self._parser or self._default_dispatch)(value)
        if isinstance(value, BaseModel):
            return value  # already-constructed instance; trust it
        return handler(value)

    def _default_dispatch(self, value: dict[str, Any]) -> Any:
        disc = self._discriminator
        if isinstance(disc, str):
            if disc not in value:
                raise ValueError(f"Missing discriminator field {disc!r}")
            tag = value[disc]
        else:
            tag = disc(value)
        member = self._registry.get(tag)
        if member is None:
            raise ValueError(
                f"Unknown tag {tag!r} for registry {self._registry.name!r}; "
                f"registered: {sorted(self._registry.members())}"
            )
        return member.model_validate(value)


_loaded_plugins: set[str] = set()


def load_plugins(modules: Iterable[str]) -> None:
    """Import each named module for its registration side effects (idempotent).

    A plugin module registers members simply by being imported (its top-level
    ``@reg.register(...)`` decorators run). Safe to call repeatedly and from any
    process. The module list is passed explicitly (from a config's ``imports``
    key, or threaded into spawn workers) -- there is deliberately no env-var
    fallback.
    """
    for name in modules:
        if name in _loaded_plugins:
            continue
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import plugin module {name!r} (declared in config "
                "`imports`). Ensure it is importable from the working directory "
                "or installed in the environment."
            ) from exc
        _loaded_plugins.add(name)
        logger.info("Loaded plugin module %s", name)


__all__ = ["Registry", "RegistryUnion", "load_plugins"]


if __name__ == "__main__":
    import json
    from typing import Literal

    from pydantic import ConfigDict
    from rich import print

    class _Base(BaseModel):
        model_config = ConfigDict(extra="forbid")
        type: str

    smoke_registry: Registry[_Base] = Registry("smoke", base=_Base)

    @smoke_registry.register("a")
    class _A(_Base):
        type: Literal["a"] = "a"
        gain: float = 1.0

    @smoke_registry.register("b")
    class _B(_Base):
        type: Literal["b"] = "b"
        width: int = 3

    Thing = Annotated[_Base, RegistryUnion(smoke_registry, "type")]

    class _Cfg(BaseModel):
        model_config = ConfigDict(extra="forbid")
        thing: Thing

    cfg = _Cfg.model_validate({"thing": {"type": "a", "gain": 2.0}})
    parsed = cfg.thing
    assert isinstance(parsed, _A) and parsed.gain == 2.0
    schema = _Cfg.model_json_schema()
    mapping = schema["properties"]["thing"]["discriminator"]["mapping"]
    assert set(mapping) == {"a", "b"}, mapping

    # Late registration validates at runtime with no model_rebuild.
    @smoke_registry.register("c")
    class _C(_Base):
        type: Literal["c"] = "c"

    late = _Cfg.model_validate({"thing": {"type": "c"}})
    assert isinstance(late.thing, _C)

    print("[green]registry smoke test passed[/green]")
    print(json.dumps(mapping, indent=2))
