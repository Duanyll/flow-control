"""Generate JSON schemas for all trainer / inference config types.

Usage:
    flow-control schema [--output-dir DIR]
"""

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter
from rich import print

from flow_control.scripts.preprocess import PreprocessConfig
from flow_control.training.grpo import GrpoTrainer
from flow_control.training.inference import Inference
from flow_control.training.nft import NftTrainer
from flow_control.training.sft import SftTrainer

DEFAULT_OUTPUT_DIR = "schema"


def generate_schemas() -> dict[str, dict]:
    """Return a mapping of config name -> JSON schema dict."""
    return {
        "sft": TypeAdapter(SftTrainer).json_schema(),
        "grpo": TypeAdapter(GrpoTrainer).json_schema(),
        "nft": TypeAdapter(NftTrainer).json_schema(),
        "inference": TypeAdapter(Inference).json_schema(),
        "preprocess": TypeAdapter(PreprocessConfig).json_schema(),
    }


def run(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Generate schemas and write them to the output directory."""
    schemas = generate_schemas()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, schema in schemas.items():
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"Wrote {path}")
    print(
        "\n[red]If your IDE caches JSON schemas (e.g. VSCode), "
        "reload the window to pick up the updated schemas.[/red]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON schemas for config types."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write schema files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
