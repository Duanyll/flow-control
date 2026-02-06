import datetime
from typing import Any

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
INDENT_STR = "  "  # Define the base indent string

# --- Helper Functions ---


def pretty_str(value: Any, limit: int = DEFAULT_STR_LIMIT) -> str:
    """Formats a value as a string, handling truncation and special characters."""
    # (No changes needed in pretty_str)
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

    # --- Basic Types ---
    # (No changes needed for basic types, strings, bytes, tensors, ndarrays)
    if value is None:
        return f"[{STYLE_NONE}]None[/]"
    if value is True:
        return f"[{STYLE_BOOL_TRUE}]True[/]"
    if value is False:
        return f"[{STYLE_BOOL_FALSE}]False[/]"
    if isinstance(value, (int, float, complex)):
        return f"[{STYLE_TYPE}]{type(value).__name__}[/] [{STYLE_VALUE}]{value}[/]"
    if isinstance(value, (datetime.datetime, datetime.date, datetime.timedelta)):
        return f"[{STYLE_TYPE}]{type(value).__name__}[/] [{STYLE_VALUE}]{value!s}[/]"
    if isinstance(value, str):
        length = len(value)
        return f"[{STYLE_TYPE}]str[/] [{STYLE_INFO}][{length}][/] {pretty_str(value, limit=str_limit)}"
    if isinstance(value, bytes):
        length = len(value)
        preview = ""
        try:
            if length <= str_limit // 3:
                preview = f" ≈ {pretty_str(value.decode('utf-8', errors='replace'), limit=str_limit)}"
        except Exception:
            pass
        return f"[{STYLE_TYPE}]bytes[/] [{STYLE_INFO}][{length}][/]{preview}"
    if isinstance(value, torch.Tensor):
        try:
            device = str(value.device)
            dtype = str(value.dtype).split(".")[-1]
            shape_str = rich.markup.escape(str(list(value.shape)))
            requires_grad_str = (
                f" [{STYLE_TENSOR}]grad[/]" if value.requires_grad else ""
            )
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
            return (
                f"[{STYLE_TENSOR}]Tensor[/] [{STYLE_ERROR}](Error describing: {e})[/]"
            )
    if isinstance(value, np.ndarray):
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
            return (
                f"[{STYLE_NDARRAY}]ndarray[/] [{STYLE_ERROR}](Error describing: {e})[/]"
            )

    # --- Pandas DataFrames/Series (Optional) ---
    if isinstance(value, pd.DataFrame):
        try:
            # *** CORRECTED INDENTATION FOR DATAFRAME ***
            shape_str = rich.markup.escape(str(value.shape))
            mem = value.memory_usage(deep=True).sum()
            mem_str = (
                f"{mem / 1024**2:.2f} MiB" if mem > 1024**2 else f"{mem / 1024:.1f} KiB"
            )
            head = f"[{STYLE_PANDAS}]DataFrame[/] [{STYLE_SHAPE}]{shape_str}[/] [{STYLE_INFO}]({len(value.columns)} cols, {mem_str})[/]"

            lines = [head]  # Title line, no indent needed here
            # Calculate indents based on current depth
            base_indent = INDENT_STR * (current_depth + 1)
            cols_indent = base_indent + INDENT_STR
            rows_indent = (
                base_indent + INDENT_STR
            )  # Row items indented one level deeper than base

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
            lines.append(
                cols_indent + ", ".join(cols_repr)
            )  # Apply indent to the joined columns string

            if len(value) > 0:
                lines.append(f"{base_indent}[{STYLE_INFO}]Head:[/]")
                seen_ids.add(value_id)
                for i, row_tuple in enumerate(value.head(max_items).itertuples()):
                    if i >= max_items:  # Redundant check, head already limits, but safe
                        break
                    row_dict = row_tuple._asdict()  # type: ignore
                    index_val = row_dict.pop("Index")
                    # Recursive call increases depth automatically
                    row_desc = _describe_recursive(
                        row_dict,
                        max_items,
                        max_depth,
                        current_depth + 1,
                        str_limit,
                        seen_ids.copy(),
                        inspect_objects,
                    )
                    # Apply indent to the row line
                    lines.append(
                        f"{rows_indent}[{STYLE_KEY}]Index {index_val}[/]: {row_desc}"
                    )

                if len(value) > max_items:
                    lines.append(
                        f"{rows_indent}[{STYLE_INFO}]... ({len(value) - max_items} more rows)[/]"
                    )  # Apply indent
                seen_ids.remove(value_id)
            return "\n".join(lines)

        except Exception as e:
            return f"[{STYLE_PANDAS}]DataFrame[/] [{STYLE_ERROR}](Error describing: {e})[/]"

    if isinstance(value, pd.Series):
        try:
            # *** CORRECTED INDENTATION FOR SERIES ***
            shape_str = rich.markup.escape(str(value.shape))
            dtype = str(value.dtype)
            name = f"'{value.name}'" if value.name else "(No Name)"
            mem = value.memory_usage(deep=True)
            mem_str = (
                f"{mem / 1024**2:.2f} MiB" if mem > 1024**2 else f"{mem / 1024:.1f} KiB"
            )
            head = f"[{STYLE_PANDAS}]Series[/] [{STYLE_INFO}]{name} {dtype}[/] [{STYLE_SHAPE}]{shape_str}[/] [{STYLE_INFO}]({mem_str})[/]"

            lines = [head]  # Title line
            item_indent_str = INDENT_STR * (current_depth + 1)  # Indent for items

            seen_ids.add(value_id)
            items_to_show = value.head(max_items).items()
            item_count = 0
            for idx, val in enumerate(items_to_show):
                # Recursive call increases depth
                val_desc = _describe_recursive(
                    val,
                    max_items,
                    max_depth,
                    current_depth + 1,
                    str_limit,
                    seen_ids.copy(),
                    inspect_objects,
                )
                # Apply indent to item line
                lines.append(
                    f"{item_indent_str}[{STYLE_KEY}]{pretty_str(idx, limit=str_limit // 2)}[/]: {val_desc}"
                )
                item_count += 1  # Use explicit counter in case iterator != head len

            if len(value) > item_count:  # More accurate check
                lines.append(
                    f"{item_indent_str}[{STYLE_INFO}]... ({len(value) - item_count} more items)[/]"
                )  # Apply indent
            seen_ids.remove(value_id)

            return "\n".join(lines)
        except Exception as e:
            return (
                f"[{STYLE_PANDAS}]Series[/] [{STYLE_ERROR}](Error describing: {e})[/]"
            )

    if isinstance(value, Image.Image):
        try:
            mode = value.mode
            size = value.size
            head = f"[{STYLE_TYPE}]PIL Image[/] [{STYLE_INFO}]{mode}[/] [{STYLE_SHAPE}]{size}[/]"
            return head
        except Exception as e:
            return (
                f"[{STYLE_TYPE}]PIL Image[/] [{STYLE_ERROR}](Error describing: {e})[/]"
            )

    # --- Collections (list, tuple, dict, set, frozenset) ---
    is_collection = False
    is_mapping = False
    items_iterator: Any = None
    collection_type_name = type(value).__name__
    collection_len = -1
    key_formatter: Any = None  # Function to format keys for display

    if isinstance(value, (list, tuple)):
        is_collection = True
        items_iterator = enumerate(value)
        collection_len = len(value)
        key_formatter = lambda k: (  # noqa: E731
            f"[{STYLE_KEY}]\\[{k}][/]"
            if isinstance(value, list)
            else f"[{STYLE_KEY}]({k})[/]"
        )
    elif isinstance(value, dict):
        is_collection = True
        is_mapping = True
        items_iterator = value.items()
        collection_len = len(value)
        key_formatter = (  # noqa: E731
            lambda k: f"[{STYLE_KEY}]{pretty_str(k, limit=str_limit // 2)}[/]"
        )
    elif isinstance(value, (set, frozenset)):
        is_collection = True
        items_iterator = enumerate(value)
        collection_len = len(value)
        key_formatter = (  # noqa: E731
            lambda k: f"[{STYLE_INFO}]-[/]"
        )  # Indicate set item

    if is_collection:
        # *** CORRECTED INDENTATION FOR COLLECTIONS ***
        title = f"[{STYLE_TYPE}]{collection_type_name}[/] [{STYLE_INFO}]with {collection_len} items[/]"
        lines = [title]  # Title line
        item_indent_str = INDENT_STR * (current_depth + 1)  # Calculate indent for items

        seen_ids.add(value_id)
        item_count = 0
        try:
            for item_or_pair in items_iterator:
                if item_count >= max_items:
                    lines.append(
                        f"{item_indent_str}[{STYLE_INFO}]... ({collection_len - max_items} more items)[/]"
                    )  # Apply indent
                    break

                if is_mapping:
                    k, v = item_or_pair
                else:
                    k, v = item_or_pair  # k is index/enum count

                key_str = key_formatter(k)
                # Recursive call increases depth
                value_desc = _describe_recursive(
                    v,
                    max_items,
                    max_depth,
                    current_depth + 1,
                    str_limit,
                    seen_ids.copy(),
                    inspect_objects,
                )
                lines.append(
                    f"{item_indent_str}{key_str}: {value_desc}"
                )  # Apply indent
                item_count += 1
        except Exception as e:
            lines.append(
                f"{item_indent_str}[{STYLE_ERROR}](Error iterating collection: {e})[/]"
            )  # Apply indent
        finally:
            seen_ids.remove(value_id)

        return "\n".join(lines)

    # --- Generic Objects ---
    try:
        obj_repr = repr(value)
    except Exception as e:
        obj_repr = f"[{STYLE_ERROR}](Error calling repr: {e})[/]"

    head = f"[{STYLE_OBJECT}]{type(value).__name__}[/]"
    if (
        not inspect_objects
        or not hasattr(value, "__dict__")
        or not getattr(value, "__dict__", None)
    ):  # Safer check for __dict__
        limit = str_limit * 2
        return f"{head} {pretty_str(obj_repr, limit=limit)}"
    else:
        # *** CORRECTED INDENTATION FOR OBJECT ATTRIBUTES ***
        attrs = value.__dict__  # Use the actual dict
        attr_len = len(attrs)
        # Combine head, repr, and attr count
        title = f"{head} {pretty_str(obj_repr, limit=str_limit)} [{STYLE_INFO}](with {attr_len} attributes)[/]"
        lines = [title]  # Title line
        item_indent_str = INDENT_STR * (
            current_depth + 1
        )  # Calculate indent for attributes

        seen_ids.add(value_id)
        item_count = 0
        try:
            for k, v in attrs.items():
                if item_count >= max_items:
                    lines.append(
                        f"{item_indent_str}[{STYLE_INFO}]... ({attr_len - max_items} more attributes)[/]"
                    )  # Apply indent
                    break

                key_str = f"[{STYLE_KEY}]{pretty_str(k, limit=str_limit // 2)}[/]"
                # Recursive call increases depth
                value_desc = _describe_recursive(
                    v,
                    max_items,
                    max_depth,
                    current_depth + 1,
                    str_limit,
                    seen_ids.copy(),
                    inspect_objects,
                )
                lines.append(
                    f"{item_indent_str}{key_str}: {value_desc}"
                )  # Apply indent
                item_count += 1
        except Exception as e:
            lines.append(
                f"{item_indent_str}[{STYLE_ERROR}](Error iterating attributes: {e})[/]"
            )  # Apply indent
        finally:
            seen_ids.remove(value_id)

        return "\n".join(lines)


# --- Public API --- (No changes needed)
def describe(
    value: Any,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    str_limit: int = DEFAULT_STR_LIMIT,
    inspect_objects: bool = False,
    console: Console | None = None,
):
    """
    Prints a rich, detailed description of a Python variable.
    (docstring unchanged)
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
    console.print(description)


# --- Example Usage --- (No changes needed, but output will be correctly indented)
def _demo():
    # (Example data structures remain the same)
    my_dict = {
        "a_string": "Hello world!\nThis is a test string with\ttabs and newlines." * 3,
        "an_int": 12345,
        "a_float": 3.1415926535,
        "a_bool": True,
        "a_none": None,
        "a_list": [
            1,
            2.0,
            "three",
            None,
            [4, 5, {"deep": True}],
        ],  # Added deeper nesting
        "a_tensor": torch.randn(5, 3, dtype=torch.float32) * 100,
        "a_bool_tensor": torch.rand(100, 100) > 0.8,
        "an_ndarray": np.arange(15).reshape(3, 5) + 0.5,
        "a_set": {1, "apple", 3.14, None, ("tuple", "inside")},
        "datetime_obj": datetime.datetime.now(),
        "large_tensor": torch.zeros(1000, 1000),
    }
    my_dict["self_ref"] = my_dict  # Cycle

    class MyObject:
        def __init__(self, name, data):
            self.name = name
            self.data = data
            self._private_ish = "secret"

        def __repr__(self):
            return f"MyObject(name='{self.name}')"

    my_object = MyObject(
        "TestObj", {"nested": True, "items": [10, 20, my_dict["a_list"]]}
    )  # Nested object data
    my_dict["custom_object"] = my_object

    nested_list = [1, [2, [3, [4, [5, [6, [7]]]]]]]

    print("--- Basic Description ---")
    describe(my_dict, max_depth=7)  # Increase depth to see nesting

    print("\n--- Deeper Inspection ---")
    describe(my_dict, max_items=5, max_depth=7, inspect_objects=True)

    print("\n--- Nested List (Depth Limit - Corrected Indentation) ---")
    describe(nested_list, max_depth=4)

    print("\n--- Pandas DataFrame ---")
    df_data = {
        "col1": [1, 2, 3, 4, 5] * 2,
        "col2": ["A", "B", "C", "D", "E"] * 2,
        "col3": np.random.rand(10) > 0.5,
        "col4_long_name_xxxxx": pd.Timestamp("20230101")
        + pd.to_timedelta(np.arange(10), "D"),
    }
    my_df = pd.DataFrame(df_data)
    my_df.loc[1, "col2"] = "A very long string value to test truncation" * 2
    # Add a nested structure within the DataFrame for testing
    my_df["nested_col"] = [list(range(i)) for i in range(10)]

    describe(
        my_df, max_items=3, str_limit=50, max_depth=5
    )  # Limit depth for DF example

    print("\n--- Pandas Series ---")
    my_series = my_df["nested_col"].copy()  # Use the nested column
    my_series.name = "My Nested Series"
    describe(my_series, max_items=4, max_depth=5)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pretty-print pickled or saved numpy/torch files to make your life easier."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the demo showcasing describe capabilities.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help="Maximum items to show in collections.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum depth for nested structures.",
    )
    parser.add_argument(
        "--str-limit",
        type=int,
        default=DEFAULT_STR_LIMIT,
        help="Character limit for string values.",
    )
    parser.add_argument(
        "--inspect-objects",
        action="store_true",
        help="Whether to inspect object attributes if __dict__ is present.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a .pkl, .npy, .npz, .pt, .pth file for describing its contents.",
    )
    args = parser.parse_args()

    if args.demo:
        _demo()
    elif args.path:
        import os
        import pickle

        path = args.path
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            return

        try:
            if path.endswith((".pkl", ".pickle")):
                with open(path, "rb") as f:
                    data = pickle.load(f)
            elif path.endswith(".npy") or path.endswith(".npz"):
                data = np.load(path, allow_pickle=True)
            elif path.endswith((".pt", ".pth")):
                data = torch.load(path, map_location="cpu")
            else:
                print(
                    "Unsupported file type. Please provide a .pkl, .npy, .npz, .pt, or .pth file."
                )
                return

            describe(
                data,
                max_items=args.max_items,
                max_depth=args.max_depth,
                str_limit=args.str_limit,
                inspect_objects=args.inspect_objects,
            )
        except Exception as e:
            print(f"Error loading or describing file: {e}")
    else:
        parser.print_help()
