import argparse
import importlib
import sys
from pathlib import Path

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

    lora_sub = subparsers.add_parser(
        "lora", help="Offline Diffusers LoRA conversion and fusion."
    )
    lora_actions = lora_sub.add_subparsers(dest="lora_action", required=True)

    def add_lora_config(action: argparse.ArgumentParser) -> None:
        action.add_argument(
            "config_path", type=str, help="Training/model configuration file."
        )
        add_config_patch_arguments(action)

    def add_external_lora(action: argparse.ArgumentParser) -> None:
        action.add_argument("--lora", required=True, help="LoRA path or Hub id.")
        action.add_argument(
            "--weight-name",
            default=None,
            help="Specific input weight file inside a directory or Hub repo.",
        )
        action.add_argument("--output-dir", required=True)

    lora_export = lora_actions.add_parser(
        "export", help="Export DCP training weights to Diffusers LoRA."
    )
    add_lora_config(lora_export)
    lora_export.add_argument("--checkpoint-dir", required=True)
    lora_export.add_argument(
        "--weights",
        choices=("current", "ema", "ema_old"),
        default="current",
    )
    lora_export.add_argument("--output-dir", required=True)
    lora_export.add_argument(
        "--weight-name", default="pytorch_lora_weights.safetensors"
    )

    lora_import = lora_actions.add_parser(
        "import", help="Convert a Diffusers-compatible LoRA to transformer-only DCP."
    )
    add_lora_config(lora_import)
    add_external_lora(lora_import)

    lora_convert = lora_actions.add_parser(
        "convert",
        help="Normalize a Diffusers-compatible LoRA through its official loader.",
    )
    add_lora_config(lora_convert)
    add_external_lora(lora_convert)
    lora_convert.add_argument(
        "--output-weight-name", default="pytorch_lora_weights.safetensors"
    )

    lora_fuse = lora_actions.add_parser(
        "fuse", help="Fuse a LoRA into a fresh Diffusers transformer on CPU."
    )
    add_lora_config(lora_fuse)
    add_external_lora(lora_fuse)
    lora_fuse.add_argument("--scale", type=float, default=1.0)

    report_sub = subparsers.add_parser(
        "report",
        help="Append a Markdown report to an existing trackio run.",
    )
    report_sub.add_argument(
        "project", type=str, help="Trackio project (== experiment_name)."
    )
    report_sub.add_argument("run_id", type=str, help="Trackio run name (== run_id).")
    report_source = report_sub.add_mutually_exclusive_group(required=True)
    report_source.add_argument(
        "--file", type=str, help="Path to a markdown file to log."
    )
    report_source.add_argument("--text", type=str, help="Markdown text logged inline.")
    report_sub.add_argument(
        "--key",
        type=str,
        default="report",
        help="Metric key to log the markdown under (default: report).",
    )
    report_sub.add_argument(
        "--step", type=int, default=None, help="Optional step to log the report at."
    )
    report_sub.add_argument(
        "--trackio-dir",
        type=str,
        default="./runs/.trackio",
        help="Trackio DB directory (default: ./runs/.trackio).",
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


def _dispatch_lora(args: argparse.Namespace) -> None:
    config = load_config_file(
        args.config_path, args.config_updates, args.config_removes
    )
    load_plugins(config.get("imports", []))

    from flow_control.scripts import lora

    if args.lora_action == "export":
        lora.export_dcp(
            config,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            checkpoint_weights=args.weights,
            weight_name=args.weight_name,
        )
    elif args.lora_action == "import":
        lora.import_dcp(
            config,
            lora_path=args.lora,
            output_dir=args.output_dir,
            weight_name=args.weight_name,
        )
    elif args.lora_action == "convert":
        lora.convert(
            config,
            lora_path=args.lora,
            output_dir=args.output_dir,
            input_weight_name=args.weight_name,
            output_weight_name=args.output_weight_name,
        )
    elif args.lora_action == "fuse":
        lora.fuse(
            config,
            lora_path=args.lora,
            output_dir=args.output_dir,
            scale=args.scale,
            weight_name=args.weight_name,
        )
    else:
        raise ValueError(f"Unknown LoRA action: {args.lora_action}")


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

    if command == "report":
        from flow_control.scripts.report import run as run_report

        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        run_report(
            args.project,
            args.run_id,
            text,
            key=args.key,
            step=args.step,
            trackio_dir=args.trackio_dir,
        )
        return

    if command == "lora":
        _dispatch_lora(args)
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
