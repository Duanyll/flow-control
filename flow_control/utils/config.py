import argparse
import contextlib
import tomllib
from collections.abc import MutableMapping, Sequence
from typing import Any

import json5
import yaml
from jsonpath_ng.ext import parse
from pydantic import BaseModel


def add_config_patch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--update",
        dest="config_updates",
        action="append",
        default=[],
        metavar="JSONPATH=VALUE",
        help=(
            "Update all config nodes matched by JSONPath. VALUE is parsed as a "
            "JSON5 literal when possible, otherwise as a string. Can be repeated."
        ),
    )
    parser.add_argument(
        "--remove",
        dest="config_removes",
        action="append",
        default=[],
        metavar="JSONPATH",
        help="Remove all config nodes matched by JSONPath. Can be repeated.",
    )


def format_config_patch_args(
    updates: Sequence[str] = (),
    removes: Sequence[str] = (),
) -> list[str]:
    """Render config patches back into ``--update``/``--remove`` CLI args.

    Used to forward overrides across the process boundary to subprocesses that
    re-load the same config file (e.g. ``launch`` spawning ``seed`` / torchrun).
    """
    args: list[str] = []
    for update in updates:
        args.extend(["--update", update])
    for remove in removes:
        args.extend(["--remove", remove])
    return args


def _find_update_separator(update: str) -> int | None:
    quote: str | None = None
    escaped = False
    bracket_depth = 0

    for i, char in enumerate(update):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "[":
            bracket_depth += 1
            continue
        if char == "]" and bracket_depth:
            bracket_depth -= 1
            continue
        if char == "=" and bracket_depth == 0:
            return i

    return None


def _split_update(update: str) -> tuple[str, str]:
    separator = _find_update_separator(update)
    if separator is None:
        raise ValueError(
            f"Config update must be formatted as JSONPATH=VALUE: {update!r}"
        )

    path = update[:separator].strip()
    value = update[separator + 1 :].strip()
    if not path:
        raise ValueError(f"Empty JSONPath in config update: {update!r}")
    return path, value


def _parse_update_value(value: str) -> Any:
    try:
        return json5.loads(value)
    except ValueError:
        return value


def _require_matches(config: Any, expression: str) -> Any:
    path = parse(expression)
    if not path.find(config):
        raise ValueError(f"JSONPath matched nothing: {expression}")
    return path


def apply_config_patches(
    config: dict[str, Any],
    updates: Sequence[str] = (),
    removes: Sequence[str] = (),
) -> dict[str, Any]:
    for update in updates:
        expression, value_text = _split_update(update)
        # update_or_create (not strict .update): lets --update SET a key that the
        # config file omits but the Pydantic model defines (e.g. run_id, attempt_id,
        # which default to None and aren't written in the jsonc). Typos that create a
        # genuinely unknown key are still caught downstream by the models'
        # extra="forbid" validation.
        parse(expression).update_or_create(config, _parse_update_value(value_text))
    for expression in removes:
        _require_matches(config, expression).filter(lambda _: True, config)
    return config


def flatten_dict(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def deep_merge_dicts(dict1, dict2):
    """
    Recursively merge dict2 into dict1.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config_file(
    path: str,
    updates: Sequence[str] = (),
    removes: Sequence[str] = (),
) -> dict:
    """
    Load JSON, YAML or TOML configuration file.
    """
    if path.endswith((".json", ".json5", ".jsonc")):
        with open(path, encoding="utf-8") as f:
            res = json5.load(f)
    elif path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as f:
            res = yaml.safe_load(f)
    elif path.endswith(".toml"):
        with open(path, "rb") as f:
            res = tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path}")

    if isinstance(res, dict) and "$schema" in res:
        del res["$schema"]

    if not isinstance(res, dict):
        raise ValueError(f"Config file must contain an object: {path}")

    apply_config_patches(res, updates, removes)
    return res


def deep_update_pydantic_model_with_dict(model: BaseModel, updates: dict[str, Any]):
    if not updates:
        return {}

    model_dump = model.model_dump(warnings="none", exclude_defaults=True)
    updated_model_dump = deep_merge_dicts(model_dump, updates)
    temp_model = model.model_validate(updated_model_dump)

    def copy_attributes(dest: BaseModel, src: BaseModel, updates: dict[str, Any]):
        backup = {}
        for key in updates:
            src_value = getattr(src, key)
            dest_value = getattr(dest, key)
            if isinstance(src_value, BaseModel) and isinstance(dest_value, BaseModel):
                backup[key] = copy_attributes(dest_value, src_value, updates[key])
            else:
                setattr(dest, key, src_value)
                backup[key] = dest_value
        return backup

    backup = copy_attributes(model, temp_model, updates)
    return backup


def deep_restore_pydantic_model_with_dict(model: BaseModel, backup: dict[str, Any]):
    if not backup:
        return

    def restore_attributes(dest: BaseModel, backup: dict[str, Any]):
        for key, value in backup.items():
            dest_value = getattr(dest, key)
            if isinstance(value, dict) and isinstance(dest_value, BaseModel):
                restore_attributes(dest_value, value)
            else:
                setattr(dest, key, value)

    restore_attributes(model, backup)


@contextlib.contextmanager
def override_pydantic_model_fields(model: BaseModel, overrides: dict[str, Any]):
    """
    Context manager to temporarily override fields of a Pydantic model. Not guaranteed
    to work with complex nested or polymorphic models.
    Args:
        model: The Pydantic model instance to override.
        overrides: A dictionary of field names and their temporary values.
    Usage:
        with override_pydantic_model_fields(model, {"field1": value1}):
            # model.field1 is temporarily set to value1
            ...
    """
    backup = deep_update_pydantic_model_with_dict(model, overrides)
    try:
        yield
    finally:
        deep_restore_pydantic_model_with_dict(model, backup)
