import datetime
import os
import pickle

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from rich import print

from flow_control.utils.describe import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITEMS,
    DEFAULT_STR_LIMIT,
    describe,
)

# --- Example Usage ---


def _demo():
    # Basic data structures
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
        ],
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
    )
    my_dict["custom_object"] = my_object

    nested_list = [1, [2, [3, [4, [5, [6, [7]]]]]]]

    print("--- Basic Description ---")
    describe(my_dict, max_items=20, max_depth=5)

    print("\n--- Deeper Inspection ---")
    describe(my_dict, max_items=20, max_depth=10, inspect_objects=True)

    print("\n--- Nested List (Depth Limit) ---")
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
    my_df["nested_col"] = [list(range(i)) for i in range(10)]

    describe(my_df, max_items=3, str_limit=50, max_depth=5)

    print("\n--- Pandas Series ---")
    my_series = my_df["nested_col"].copy()
    my_series.name = "My Nested Series"
    describe(my_series, max_items=4, max_depth=5)

    # --- Pydantic v2 Example ---
    print("\n--- Pydantic v2 BaseModel ---")

    class Address(BaseModel):
        street: str
        city: str
        zip_code: str = Field(alias="zipCode")

    class Person(BaseModel):
        name: str
        age: int
        email: str | None = None
        address: Address | None = None
        tags: list[str] = []

    # Simple model
    person = Person(
        name="Alice",
        age=30,
        email="alice@example.com",
        tags=["developer", "python"],
    )
    print("Simple Pydantic model:")
    describe(person)

    # Nested model
    print("\nNested Pydantic model:")
    person_with_address = Person(
        name="Bob",
        age=25,
        email="bob@example.com",
        address=Address(street="123 Main St", city="Springfield", zipCode="12345"),
        tags=["manager", "team-lead", "agile"],
    )
    describe(person_with_address, max_depth=4)

    # List of models
    print("\nList of Pydantic models:")
    people = [
        Person(name="Charlie", age=35),
        Person(name="Diana", age=28, email="diana@example.com"),
    ]
    describe(people, max_depth=3)


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


if __name__ == "__main__":
    main()
