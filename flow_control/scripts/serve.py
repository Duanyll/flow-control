"""Start an interactive Gradio web server for testing trained models."""

import socket
import subprocess

from flow_control.serving import ServeConfig, ServingEngine, create_gradio_app
from flow_control.utils.config import load_config_file
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


def _log_node_addresses(port: int) -> None:
    """Print hostname and IPs so Slurm users know where to connect."""
    hostname = socket.gethostname()
    try:
        ips = subprocess.check_output(["hostname", "-I"], text=True, timeout=5).split()
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning(f"Could not resolve node IPs via `hostname -I`: {e}")
        ips = []

    logger.info(f"Serving node hostname: {hostname}")
    if ips:
        logger.info(f"Reachable at: {', '.join(f'{ip}:{port}' for ip in ips)}")


def run(config_path: str) -> None:
    config = ServeConfig(**load_config_file(config_path))

    logger.info("Initializing serving engine...")
    engine = ServingEngine(config)
    engine.load_all()

    logger.info(f"Starting Gradio server on {config.host}:{config.port}")
    _log_node_addresses(config.port)
    app = create_gradio_app(engine)
    app.launch(
        server_name=config.host,
        server_port=config.port,
        share=config.share,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start interactive Gradio serving.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    run(args.config_path)


if __name__ == "__main__":
    main()
