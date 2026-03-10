import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="flow-control",
        description="Training utilities for flow-matching Diffusion Transformers (DiTs).",
    )
    subparsers = parser.add_subparsers(dest="command")

    for name in [
        "preprocess",
        "preprocess-ray",
        "seed",
        "launch",
        "vae-server",
        "reward-server",
    ]:
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "config_path", type=str, help="Path to the configuration file."
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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    _dispatch(args.command, args)


def _dispatch(command: str, args: argparse.Namespace) -> None:
    """Lazy-import and run the appropriate subcommand."""
    if command == "schema":
        from flow_control.scripts.schema import run as run_schema

        run_schema(**({"output_dir": args.output_dir} if args.output_dir else {}))
        return

    if command == "preprocess":
        from flow_control.scripts.preprocess import run
    elif command == "preprocess-ray":
        from flow_control.scripts.preprocess_ray import run
    elif command == "seed":
        from flow_control.scripts.seed import run
    elif command == "launch":
        from flow_control.scripts.launch import run
    elif command == "vae-server":
        from flow_control.scripts.vae_server import run
    elif command == "reward-server":
        from flow_control.scripts.reward_server import run
    else:
        raise ValueError(f"Unknown command: {command}")

    run(args.config_path)


if __name__ == "__main__":
    main()
