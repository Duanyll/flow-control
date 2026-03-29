"""Start an interactive Gradio web server for testing trained models."""

from flow_control.serving import ServeConfig, ServingEngine, create_gradio_app
from flow_control.utils.config import load_config_file
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


def run(config_path: str) -> None:
    config = ServeConfig(**load_config_file(config_path))

    logger.info("Initializing serving engine...")
    engine = ServingEngine(config)
    engine.load_all()

    logger.info(f"Starting Gradio server on {config.host}:{config.port}")
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
