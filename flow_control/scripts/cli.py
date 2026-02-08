import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="flow-control",
        description="Training utilities for flow-matching Diffusion Transformers (DiTs).",
    )
    subparsers = parser.add_subparsers(dest="command")

    for name in ["preprocess", "generate-dcp-seed", "launch", "vae-server"]:
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "config_path", type=str, help="Path to the configuration file."
        )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    _dispatch(args.command, args.config_path)


def _dispatch(command: str, config_path: str) -> None:
    """Lazy-import and run the appropriate subcommand."""
    if command == "preprocess":
        from flow_control.scripts.preprocess import run
    elif command == "generate-dcp-seed":
        from flow_control.scripts.generate_dcp_seed import run
    elif command == "launch":
        from flow_control.scripts.launch import run
    elif command == "vae-server":
        from flow_control.scripts.vae_server import run
    else:
        raise ValueError(f"Unknown command: {command}")

    run(config_path)


if __name__ == "__main__":
    main()
