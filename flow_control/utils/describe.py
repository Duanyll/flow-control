import datetime
import functools
from collections.abc import Callable, Iterable
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import rich
import rich.markup
import torch
from PIL import Image
from rich.console import Console

# --- Configuration ---
DEFAULT_STR_LIMIT = 70
DEFAULT_MAX_ITEMS = 10
DEFAULT_MAX_DEPTH = 5
STYLE_TYPE = "bold light_sea_green"
STYLE_KEY = "cyan"
STYLE_VALUE = "white"
STYLE_SPECIAL_CHAR = "red"
STYLE_INFO = "dim"
STYLE_TENSOR = "dark_orange3"
STYLE_NDARRAY = "cornflower_blue"
STYLE_PANDAS = "purple4"
STYLE_SHAPE = "turquoise2"
STYLE_BOOL_TRUE = "green"
STYLE_BOOL_FALSE = "red"
STYLE_NONE = "grey50"
STYLE_ERROR = "bold red"
STYLE_CYCLE = "italic yellow"
STYLE_OBJECT = "yellow"
STYLE_PYDANTIC = "magenta"
INDENT_STR = "  "


# --- Helper Functions ---


def pretty_str(value: Any, limit: int = DEFAULT_STR_LIMIT) -> str:
    """Formats a value as a string, handling truncation and special characters."""
    if not isinstance(value, str):
        try:
            value_str = repr(value)
        except Exception:
            value_str = f"<{type(value).__name__} (repr failed)>"
    else:
        value_str = value

    original_length = len(value_str)
    truncated = False
    if original_length > limit:
        value_str = value_str[:limit]
        truncated = True

    value_str = rich.markup.escape(value_str)
    value_str = value_str.replace("\n", f"[{STYLE_SPECIAL_CHAR}]↵[/]")
    value_str = value_str.replace("\t", f"[{STYLE_SPECIAL_CHAR}]→[/]")

    if truncated:
        value_str += f"[{STYLE_SPECIAL_CHAR}]…[/]"

    if isinstance(value, str):
        return f'"{value_str}"'
    else:
        if isinstance(value, (int, float, bool)) or value is None:
            return value_str
        else:
            return value_str


# --- Type-specific Handlers ---


def _describe_none() -> str:
    return f"[{STYLE_NONE}]None[/]"


def _describe_bool(value: bool) -> str:
    if value:
        return f"[{STYLE_BOOL_TRUE}]True[/]"
    return f"[{STYLE_BOOL_FALSE}]False[/]"


def _describe_number(value: int | float | complex) -> str:
    return f"[{STYLE_TYPE}]{type(value).__name__}[/] [{STYLE_VALUE}]{value}[/]"


def _describe_datetime(
    value: datetime.datetime | datetime.date | datetime.timedelta,
) -> str:
    return f"[{STYLE_TYPE}]{type(value).__name__}[/] [{STYLE_VALUE}]{value!s}[/]"


def _describe_string(value: str, str_limit: int) -> str:
    length = len(value)
    return f"[{STYLE_TYPE}]str[/] [{STYLE_INFO}][{length}][/] {pretty_str(value, limit=str_limit)}"


def _describe_bytes(value: bytes, str_limit: int) -> str:
    length = len(value)
    preview = ""
    try:
        if length <= str_limit // 3:
            preview = f" ≈ {pretty_str(value.decode('utf-8', errors='replace'), limit=str_limit)}"
    except Exception:
        pass
    return f"[{STYLE_TYPE}]bytes[/] [{STYLE_INFO}][{length}][/]{preview}"


def _describe_tensor(value: torch.Tensor) -> str:
    try:
        device = str(value.device)
        dtype = str(value.dtype).split(".")[-1]
        shape_str = rich.markup.escape(str(list(value.shape)))
        requires_grad_str = f" [{STYLE_TENSOR}]grad[/]" if value.requires_grad else ""
        head = f"[{STYLE_TENSOR}]Tensor[/] [{STYLE_INFO}]on {device} {dtype}[/] [{STYLE_SHAPE}]{shape_str}[/] {requires_grad_str}"
        size = value.numel()
        if size == 0:
            return f"{head} [{STYLE_INFO}](empty)[/]"
        if size == 1:
            return f"{head} [{STYLE_VALUE}]{value.item()}[/]"
        if value.dtype == torch.bool:
            ratio = value.float().mean().item() * 100
            return f"{head} [{STYLE_INFO}]({ratio:.1f}% True)[/]"
        else:
            value_float = value.float()
            val_min = torch.min(value_float).item()
            val_max = torch.max(value_float).item()
            val_mean = torch.mean(value_float).item()
            val_std = torch.std(value_float).item()
            return f"{head} [{STYLE_INFO}]({val_min:.2g} .. {val_max:.2g} | μ={val_mean:.2g} ± σ={val_std:.2g})[/]"
    except Exception as e:
        return f"[{STYLE_TENSOR}]Tensor[/] [{STYLE_ERROR}](Error describing: {e})[/]"


def _describe_ndarray(value: np.ndarray) -> str:
    try:
        shape_str = rich.markup.escape(str(list(value.shape)))
        dtype = str(value.dtype)
        head = f"[{STYLE_NDARRAY}]ndarray[/] [{STYLE_INFO}]{dtype}[/] [{STYLE_SHAPE}]{shape_str}[/]"
        size = value.size
        if size == 0:
            return f"{head} [{STYLE_INFO}](empty)[/]"
        if size == 1:
            return f"{head} [{STYLE_VALUE}]{value.item()}[/]"
        if value.dtype == np.bool_:
            ratio = value.astype(float).mean() * 100
            return f"{head} [{STYLE_INFO}]({ratio:.1f}% True)[/]"
        if np.issubdtype(value.dtype, np.number):
            val_min = np.min(value)
            val_max = np.max(value)
            val_mean = np.mean(value)
            val_std = np.std(value)
            return f"{head} [{STYLE_INFO}]({val_min:.2g} .. {val_max:.2g} | μ={val_mean:.2g} ± σ={val_std:.2g})[/]"
        else:
            return f"{head} [{STYLE_INFO}](non-numeric data)[/]"
    except Exception as e:
        return f"[{STYLE_NDARRAY}]ndarray[/] [{STYLE_ERROR}](Error describing: {e})[/]"


def _describe_pil_image(value: Image.Image) -> str:
    try:
        mode = value.mode
        size = value.size
        return f"[{STYLE_TYPE}]PIL Image[/] [{STYLE_INFO}]{mode}[/] [{STYLE_SHAPE}]{size}[/]"
    except Exception as e:
        return f"[{STYLE_TYPE}]PIL Image[/] [{STYLE_ERROR}](Error describing: {e})[/]"


def _describe_dataframe(
    value: pd.DataFrame,
    max_items: int,
    max_depth: int,
    current_depth: int,
    str_limit: int,
    seen_ids: set[int],
    inspect_objects: bool,
    recurse_fn: Callable,
) -> str:
    try:
        value_id = id(value)
        shape_str = rich.markup.escape(str(value.shape))
        mem = value.memory_usage(deep=True).sum()
        mem_str = (
            f"{mem / 1024**2:.2f} MiB" if mem > 1024**2 else f"{mem / 1024:.1f} KiB"
        )
        head = f"[{STYLE_PANDAS}]DataFrame[/] [{STYLE_SHAPE}]{shape_str}[/] [{STYLE_INFO}]({len(value.columns)} cols, {mem_str})[/]"

        lines = [head]
        base_indent = INDENT_STR * (current_depth + 1)
        cols_indent = base_indent + INDENT_STR
        rows_indent = base_indent + INDENT_STR

        # Columns section
        lines.append(f"{base_indent}[{STYLE_INFO}]Columns:[/]")
        cols_repr = []
        for i, (col, dtype) in enumerate(value.dtypes.items()):
            if i >= max_items * 2:
                cols_repr.append(f"[{STYLE_INFO}]...[/]")
                break
            cols_repr.append(
                f"[{STYLE_KEY}]{pretty_str(col, limit=str_limit // 2)}[/]: [{STYLE_INFO}]{dtype}[/]"
            )
        lines.append(cols_indent + ", ".join(cols_repr))

        if len(value) > 0:
            lines.append(f"{base_indent}[{STYLE_INFO}]Head:[/]")
            seen_ids.add(value_id)
            for i, row_tuple in enumerate(value.head(max_items).itertuples()):
                if i >= max_items:
                    break
                row_dict = row_tuple._asdict()  # type: ignore
                index_val = row_dict.pop("Index")
                row_desc = recurse_fn(
                    row_dict,
                    max_items,
                    max_depth,
                    current_depth + 1,
                    str_limit,
                    seen_ids.copy(),
                    inspect_objects,
                )
                lines.append(
                    f"{rows_indent}[{STYLE_KEY}]Index {index_val}[/]: {row_desc}"
                )

            if len(value) > max_items:
                lines.append(
                    f"{rows_indent}[{STYLE_INFO}]... ({len(value) - max_items} more rows)[/]"
                )
            seen_ids.remove(value_id)
        return "\n".join(lines)

    except Exception as e:
        return f"[{STYLE_PANDAS}]DataFrame[/] [{STYLE_ERROR}](Error describing: {e})[/]"


def _describe_series(
    value: pd.Series,
    max_items: int,
    max_depth: int,
    current_depth: int,
    str_limit: int,
    seen_ids: set[int],
    inspect_objects: bool,
    recurse_fn: Callable,
) -> str:
    try:
        value_id = id(value)
        shape_str = rich.markup.escape(str(value.shape))
        dtype = str(value.dtype)
        name = f"'{value.name}'" if value.name else "(No Name)"
        mem = value.memory_usage(deep=True)
        mem_str = (
            f"{mem / 1024**2:.2f} MiB" if mem > 1024**2 else f"{mem / 1024:.1f} KiB"
        )
        head = f"[{STYLE_PANDAS}]Series[/] [{STYLE_INFO}]{name} {dtype}[/] [{STYLE_SHAPE}]{shape_str}[/] [{STYLE_INFO}]({mem_str})[/]"

        lines = [head]
        item_indent_str = INDENT_STR * (current_depth + 1)

        seen_ids.add(value_id)
        items_to_show = value.head(max_items).items()
        item_count = 0
        for idx, val in enumerate(items_to_show):
            val_desc = recurse_fn(
                val,
                max_items,
                max_depth,
                current_depth + 1,
                str_limit,
                seen_ids.copy(),
                inspect_objects,
            )
            lines.append(
                f"{item_indent_str}[{STYLE_KEY}]{pretty_str(idx, limit=str_limit // 2)}[/]: {val_desc}"
            )
            item_count += 1

        if len(value) > item_count:
            lines.append(
                f"{item_indent_str}[{STYLE_INFO}]... ({len(value) - item_count} more items)[/]"
            )
        seen_ids.remove(value_id)

        return "\n".join(lines)
    except Exception as e:
        return f"[{STYLE_PANDAS}]Series[/] [{STYLE_ERROR}](Error describing: {e})[/]"


def _describe_pydantic_model(
    value: Any,
    max_items: int,
    max_depth: int,
    current_depth: int,
    str_limit: int,
    seen_ids: set[int],
    inspect_objects: bool,
    recurse_fn: Callable,
) -> str:
    """Describe a Pydantic v2 BaseModel with field type annotations."""
    try:
        value_id = id(value)
        model_class = type(value)
        model_name = model_class.__name__

        # Get model fields info from class (not instance) to avoid deprecation warning
        model_fields = model_class.model_fields
        field_count = len(model_fields)

        head = f"[{STYLE_PYDANTIC}]{model_name}[/] [{STYLE_INFO}](Pydantic model with {field_count} fields)[/]"
        lines = [head]
        item_indent_str = INDENT_STR * (current_depth + 1)

        seen_ids.add(value_id)

        for item_count, (field_name, field_info) in enumerate(model_fields.items()):
            if item_count >= max_items:
                lines.append(
                    f"{item_indent_str}[{STYLE_INFO}]... ({field_count - max_items} more fields)[/]"
                )
                break

            # Get field type annotation
            field_type = field_info.annotation
            type_name = getattr(field_type, "__name__", str(field_type))

            # Get actual value
            field_value = getattr(value, field_name, None)
            value_desc = recurse_fn(
                field_value,
                max_items,
                max_depth,
                current_depth + 1,
                str_limit,
                seen_ids.copy(),
                inspect_objects,
            )

            # Format: field_name (Type): value
            lines.append(
                f"{item_indent_str}[{STYLE_KEY}]{field_name}[/] [{STYLE_INFO}]({type_name})[/]: {value_desc}"
            )

        seen_ids.remove(value_id)
        return "\n".join(lines)

    except Exception as e:
        return f"[{STYLE_PYDANTIC}]Pydantic Model[/] [{STYLE_ERROR}](Error describing: {e})[/]"


def _format_list_key(k: int) -> str:
    return f"[{STYLE_KEY}]\\[{k}][/]"


def _format_tuple_key(k: int) -> str:
    return f"[{STYLE_KEY}]({k})[/]"


def _format_dict_key(k: Any, str_limit: int) -> str:
    return f"[{STYLE_KEY}]{pretty_str(k, limit=str_limit // 2)}[/]"


def _format_set_key(_: int) -> str:
    return f"[{STYLE_INFO}]-[/]"


_COLLECTION_KEY_FORMATTERS: dict[type, Callable] = {
    list: _format_list_key,
    tuple: _format_tuple_key,
    set: _format_set_key,
    frozenset: _format_set_key,
}


def _describe_collection(
    value: list | tuple | dict | set | frozenset,
    max_items: int,
    max_depth: int,
    current_depth: int,
    str_limit: int,
    seen_ids: set[int],
    inspect_objects: bool,
    recurse_fn: Callable,
) -> str:
    """Describe list, tuple, dict, set, or frozenset."""
    value_id = id(value)
    collection_type_name = type(value).__name__
    collection_len = len(value)

    # Determine iterator and key formatter based on type
    # All iterators yield (key, value) pairs
    if isinstance(value, dict):
        items_iterator: Iterable[tuple[Any, Any]] = value.items()
        key_formatter: Callable = functools.partial(
            _format_dict_key, str_limit=str_limit
        )
    else:
        items_iterator = enumerate(value)
        key_formatter = _COLLECTION_KEY_FORMATTERS[type(value)]

    title = f"[{STYLE_TYPE}]{collection_type_name}[/] [{STYLE_INFO}]with {collection_len} items[/]"
    lines = [title]
    item_indent_str = INDENT_STR * (current_depth + 1)

    seen_ids.add(value_id)
    item_count = 0
    try:
        for k, v in items_iterator:
            if item_count >= max_items:
                lines.append(
                    f"{item_indent_str}[{STYLE_INFO}]... ({collection_len - max_items} more items)[/]"
                )
                break

            key_str = key_formatter(k)
            value_desc = recurse_fn(
                v,
                max_items,
                max_depth,
                current_depth + 1,
                str_limit,
                seen_ids.copy(),
                inspect_objects,
            )
            lines.append(f"{item_indent_str}{key_str}: {value_desc}")
            item_count += 1
    except Exception as e:
        lines.append(
            f"{item_indent_str}[{STYLE_ERROR}](Error iterating collection: {e})[/]"
        )
    finally:
        seen_ids.remove(value_id)

    return "\n".join(lines)


def _describe_generic_object(
    value: Any,
    max_items: int,
    max_depth: int,
    current_depth: int,
    str_limit: int,
    seen_ids: set[int],
    inspect_objects: bool,
    recurse_fn: Callable,
) -> str:
    """Describe a generic object."""
    value_id = id(value)

    try:
        obj_repr = repr(value)
    except Exception as e:
        obj_repr = f"[{STYLE_ERROR}](Error calling repr: {e})[/]"

    head = f"[{STYLE_OBJECT}]{type(value).__name__}[/]"

    if (
        not inspect_objects
        or not hasattr(value, "__dict__")
        or not getattr(value, "__dict__", None)
    ):
        limit = str_limit * 2
        return f"{head} {pretty_str(obj_repr, limit=limit)}"

    # Inspect object attributes
    attrs = value.__dict__
    attr_len = len(attrs)
    title = f"{head} {pretty_str(obj_repr, limit=str_limit)} [{STYLE_INFO}](with {attr_len} attributes)[/]"
    lines = [title]
    item_indent_str = INDENT_STR * (current_depth + 1)

    seen_ids.add(value_id)
    item_count = 0
    try:
        for k, v in attrs.items():
            if item_count >= max_items:
                lines.append(
                    f"{item_indent_str}[{STYLE_INFO}]... ({attr_len - max_items} more attributes)[/]"
                )
                break

            key_str = f"[{STYLE_KEY}]{pretty_str(k, limit=str_limit // 2)}[/]"
            value_desc = recurse_fn(
                v,
                max_items,
                max_depth,
                current_depth + 1,
                str_limit,
                seen_ids.copy(),
                inspect_objects,
            )
            lines.append(f"{item_indent_str}{key_str}: {value_desc}")
            item_count += 1
    except Exception as e:
        lines.append(
            f"{item_indent_str}[{STYLE_ERROR}](Error iterating attributes: {e})[/]"
        )
    finally:
        seen_ids.remove(value_id)

    return "\n".join(lines)


# --- Main Recursive Function ---


def _is_pydantic_model(value: Any) -> bool:
    """Check if value is a Pydantic v2 BaseModel instance."""
    try:
        from pydantic import BaseModel

        return isinstance(value, BaseModel)
    except ImportError:
        return False


def _describe_leaf_type(value: Any, str_limit: int) -> str | None:
    """Try to describe value as a simple leaf type. Returns None if not a leaf."""
    if value is None:
        return _describe_none()
    if isinstance(value, bool):
        return _describe_bool(value)
    if isinstance(value, (int, float, complex)):
        return _describe_number(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.timedelta)):
        return _describe_datetime(value)
    if isinstance(value, str):
        return _describe_string(value, str_limit)
    if isinstance(value, bytes):
        return _describe_bytes(value, str_limit)
    if isinstance(value, torch.Tensor):
        return _describe_tensor(value)
    if isinstance(value, np.ndarray):
        return _describe_ndarray(value)
    if isinstance(value, Image.Image):
        return _describe_pil_image(value)
    return None


def _describe_recursive(
    value: Any,
    max_items: int,
    max_depth: int,
    current_depth: int,
    str_limit: int,
    seen_ids: set[int],
    inspect_objects: bool = False,
) -> str:
    """Recursive core of the describe function."""
    value_id = id(value)
    if value_id in seen_ids:
        return f"[{STYLE_CYCLE}](Cycle Detected)[/]"

    if current_depth > max_depth:
        return f"[{STYLE_INFO}]... (Max Depth Reached)[/]"

    # --- Leaf Types (no recursion needed) ---
    leaf = _describe_leaf_type(value, str_limit)
    if leaf is not None:
        return leaf

    # --- Recursive Types ---
    recurse_args = (
        max_items,
        max_depth,
        current_depth,
        str_limit,
        seen_ids,
        inspect_objects,
        _describe_recursive,
    )

    if isinstance(value, pd.DataFrame):
        return _describe_dataframe(value, *recurse_args)
    if isinstance(value, pd.Series):
        return _describe_series(value, *recurse_args)
    if _is_pydantic_model(value):
        return _describe_pydantic_model(value, *recurse_args)
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return _describe_collection(value, *recurse_args)

    return _describe_generic_object(value, *recurse_args)


# --- Public API ---


@overload
def describe(
    value: Any,
    console: Console | None,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    str_limit: int = DEFAULT_STR_LIMIT,
    inspect_objects: bool = False,
) -> None: ...


@overload
def describe(
    value: Any,
    console: Literal[False],
    max_items: int = DEFAULT_MAX_ITEMS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    str_limit: int = DEFAULT_STR_LIMIT,
    inspect_objects: bool = False,
) -> str: ...


def describe(
    value: Any,
    console: Console | None | Literal[False] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    str_limit: int = DEFAULT_STR_LIMIT,
    inspect_objects: bool = False,
):
    """
    Prints a rich, detailed description of a Python variable.

    Args:
        value: The value to describe.
        max_items: Maximum items to show in collections.
        max_depth: Maximum depth for nested structures.
        str_limit: Character limit for string values.
        inspect_objects: Whether to inspect object attributes if __dict__ is present.
        console: Optional Rich Console instance.
    """
    if console is None:
        console = Console()

    seen_ids: set[int] = set()
    description = _describe_recursive(
        value,
        max_items=max_items,
        max_depth=max_depth,
        current_depth=0,
        str_limit=str_limit,
        seen_ids=seen_ids,
        inspect_objects=inspect_objects,
    )
    if console is not False:
        console.print(description)
    else:
        return description
