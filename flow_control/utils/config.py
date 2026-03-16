import contextlib
import tomllib
from collections.abc import MutableMapping
from typing import Any

import json5
import yaml
from pydantic import BaseModel


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


def load_config_file(path: str) -> dict:
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
