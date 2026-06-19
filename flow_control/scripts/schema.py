"""Generate JSON schemas for all trainer / inference config types.

Usage:
    flow-control schema [--output-dir DIR]
"""

import json
from pathlib import Path

from pydantic import TypeAdapter
from rich import print

DEFAULT_OUTPUT_DIR = "schema"


def _inject_schema_field(schema_dict: dict) -> dict:
    if "properties" not in schema_dict:
        schema_dict["properties"] = {}

    schema_dict["properties"]["$schema"] = {
        "title": "JSON Schema",
        "description": "VS Code schema declaration.",
        "type": "string",
    }
    # VSCode exclusive field to allow trailing commas in JSON files
    schema_dict["allowTrailingCommas"] = True
    return schema_dict


def generate_schemas() -> dict[str, dict]:
    """Return a mapping of config name -> JSON schema dict.

    The config classes are imported lazily here (not at module top) because
    importing them builds and caches their pydantic core schema, which freezes
    the registry-backed unions. Any plugins must be loaded before this call so
    their members are captured; see ``run``.
    """
    from flow_control.scripts.preprocess import PreprocessConfig
    from flow_control.serving.config import ServeConfig
    from flow_control.training.grpo import GrpoTrainer
    from flow_control.training.inference import Inference
    from flow_control.training.nft import NftTrainer
    from flow_control.training.sft import SftTrainer
    from flow_control.training.vae import VaeTrainer

    schemas = {
        "sft": TypeAdapter(SftTrainer).json_schema(),
        "grpo": TypeAdapter(GrpoTrainer).json_schema(),
        "nft": TypeAdapter(NftTrainer).json_schema(),
        "inference": TypeAdapter(Inference).json_schema(),
        "preprocess": TypeAdapter(PreprocessConfig).json_schema(),
        "vae": TypeAdapter(VaeTrainer).json_schema(),
        "serve": TypeAdapter(ServeConfig).json_schema(),
    }
    return {name: _inject_schema_field(schema) for name, schema in schemas.items()}


def run(output_dir: str = DEFAULT_OUTPUT_DIR, config_path: str | None = None) -> None:
    """Generate schemas and write them to the output directory.

    If ``config_path`` is given, its ``imports`` plugin modules are loaded
    before the config classes are imported/schema-built, so the emitted schemas
    include that config's plugin members. With no config the core-only schemas
    are emitted.
    """
    if config_path is not None:
        from flow_control.utils.config import load_config_file
        from flow_control.utils.registry import load_plugins

        config = load_config_file(config_path)
        load_plugins(config.get("imports", []))

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
