import argparse
import importlib
import sys

from flow_control.utils.config import add_config_patch_arguments, load_config_file
from flow_control.utils.registry import load_plugins

# Subcommands that take a single config file and whose ``run(config)`` consumes
# the already-loaded dict. ``launch`` also takes a config file but is dispatched
# specially (it re-spawns subprocesses that re-load the file), so it is not here.
CONFIG_COMMAND_MODULES = {
    "preprocess": "flow_control.scripts.preprocess",
    "seed": "flow_control.scripts.seed",
    "vae-server": "flow_control.scripts.vae_server",
    "reward-server": "flow_control.scripts.reward_server",
    "serve": "flow_control.scripts.serve",
}


def main():
    parser = argparse.ArgumentParser(
        prog="flow-control",
        description="Training utilities for flow-matching Diffusion Transformers (DiTs).",
    )
    subparsers = parser.add_subparsers(dest="command")

    for name in (*CONFIG_COMMAND_MODULES, "launch"):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "config_path", type=str, help="Path to the configuration file."
        )
        add_config_patch_arguments(sub)

    export_sub = subparsers.add_parser(
        "export", help="Export DCP checkpoints to HuggingFace format."
    )
    export_sub.add_argument(
        "config_path", type=str, help="Path to the training configuration file."
    )
    add_config_patch_arguments(export_sub)
    export_sub.add_argument(
        "--output-dir", type=str, required=True, help="Output directory."
    )
    export_sub.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="DCP checkpoint directory. Defaults to latest step.",
    )

    schema_sub = subparsers.add_parser(
        "schema", help="Generate JSON schemas for config types."
    )
    schema_sub.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write schema files (default: schema).",
    )
    schema_sub.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config file whose `imports` plugins are loaded before "
            "building schemas, so emitted schemas include those plugin members."
        ),
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    _dispatch(args)


def _dispatch(args: argparse.Namespace) -> None:
    """Lazy-import and run the appropriate subcommand."""
    command = args.command

    if command == "schema":
        from flow_control.scripts.schema import run as run_schema

        kwargs: dict[str, str] = {}
        if args.output_dir:
            kwargs["output_dir"] = args.output_dir
        if args.config:
            kwargs["config_path"] = args.config
        run_schema(**kwargs)
        return

    # ``launch`` re-spawns subprocesses that re-load the config file themselves,
    # so it needs the path and the raw patch args rather than a loaded dict.
    if command == "launch":
        from flow_control.scripts.launch import run as run_launch

        run_launch(args.config_path, args.config_updates, args.config_removes)
        return

    config = load_config_file(
        args.config_path, args.config_updates, args.config_removes
    )
    # Import any declared plugin modules for their registry side effects BEFORE
    # constructing a config (validation reads the registries) or dispatching.
    load_plugins(config.get("imports", []))

    if command == "export":
        from flow_control.scripts.export import run as run_export

        run_export(config, args.output_dir, args.checkpoint_dir)
        return

    module_name = CONFIG_COMMAND_MODULES.get(command)
    if module_name is None:
        raise ValueError(f"Unknown command: {command}")

    importlib.import_module(module_name).run(config)


if __name__ == "__main__":
    main()
