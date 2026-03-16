import asyncio
from typing import cast

import torch
import uvicorn
from pydantic import BaseModel, ConfigDict, TypeAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from flow_control.processors.components.vae import VAE, BaseVAE
from flow_control.utils.logging import get_logger
from flow_control.utils.remote import deserialize_tensor, serialize_tensor
from flow_control.utils.tensor import load_config_file
from flow_control.utils.types import TorchDevice

logger = get_logger(__name__)

_vae_ta = TypeAdapter(VAE)


class VAEServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 8000
    device: TorchDevice = torch.device("cuda")

    vae: VAE | None = None


def create_app(config: VAEServerConfig) -> Starlette:
    gpu_lock = asyncio.Lock()
    model_lock = asyncio.Lock()

    state: dict[str, BaseVAE | dict | None] = {
        "vae": None,
        "config_json": None,
    }

    # Pre-load if provided in config
    if config.vae is not None:
        config.vae.load_model(config.device)
        state["vae"] = config.vae
        state["config_json"] = config.vae.model_dump(mode="json", exclude={"endpoint"})

    def _get_vae() -> BaseVAE:
        vae = state["vae"]
        if vae is None:
            raise RuntimeError("No VAE model loaded. POST to /load first.")
        return vae  # type: ignore[return-value]

    async def load(request: Request) -> Response:
        body = await request.json()
        async with model_lock:
            if state["config_json"] == body:
                logger.info("VAE config unchanged, skipping reload.")
                return JSONResponse({"status": "ok", "loaded": False})
            logger.info(f"Loading VAE from client config: {body.get('type', '?')}")
            vae = cast(BaseVAE, _vae_ta.validate_python(body))
            vae.load_model(config.device)
            state["vae"] = vae
            state["config_json"] = body
            logger.info("VAE loaded successfully.")
            return JSONResponse({"status": "ok", "loaded": True})

    async def encode(request: Request) -> Response:
        vae = _get_vae()
        body = await request.body()
        images = deserialize_tensor(body, config.device, torch.bfloat16)
        logger.info(f"Encoding images with shape {images.shape}")
        async with gpu_lock:
            with torch.no_grad():
                latents = vae._encode(images)
            result = serialize_tensor(latents)
        return Response(content=result, media_type="application/octet-stream")

    async def decode(request: Request) -> Response:
        vae = _get_vae()
        body = await request.body()
        latents = deserialize_tensor(body, config.device, torch.bfloat16)
        logger.info(f"Decoding latents with shape {latents.shape}")
        async with gpu_lock:
            with torch.no_grad():
                images = vae._decode(latents)
            result = serialize_tensor(images)
        return Response(content=result, media_type="application/octet-stream")

    async def health(request: Request) -> Response:
        vae = state["vae"]
        if vae is None:
            return JSONResponse({"status": "ok", "loaded": False})
        assert isinstance(vae, BaseVAE)
        return JSONResponse(
            {
                "status": "ok",
                "loaded": True,
                "pretrained_model_id": vae.pretrained_model_id,
                "revision": vae.revision,
                "subfolder": vae.subfolder,
            }
        )

    routes = [
        Route("/load", load, methods=["POST"]),
        Route("/encode", encode, methods=["POST"]),
        Route("/decode", decode, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ]

    return Starlette(routes=routes)


def run(config_path: str) -> None:
    """Start the VAE server with the given config file."""
    config = VAEServerConfig(**load_config_file(config_path))

    if config.vae is not None:
        logger.info(f"Pre-loading VAE model: {config.vae.pretrained_model_id}")

    logger.info(f"Starting VAE server on {config.host}:{config.port}")
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start VAE encoding/decoding server.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    run(args.config_path)


if __name__ == "__main__":
    main()
