import asyncio
import io
import pickle
import sys

import torch
import uvicorn
from pydantic import BaseModel, ConfigDict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from flow_control.processors.components.vae import VAE, BaseVAE
from flow_control.utils.common import load_config_file
from flow_control.utils.logging import get_logger
from flow_control.utils.types import TorchDevice

logger = get_logger(__name__)


class VAEServerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    host: str = "0.0.0.0"
    port: int = 8000
    device: TorchDevice = torch.device("cuda")

    vae: VAE


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes, converting to bf16 first."""
    tensor = tensor.to(torch.bfloat16).cpu()
    buffer = io.BytesIO()
    pickle.dump(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes, device: torch.device) -> torch.Tensor:
    """Deserialize tensor from bytes."""
    buffer = io.BytesIO(data)
    tensor = pickle.load(buffer)  # noqa: S301
    return tensor.to(device=device)


def create_app(vae: BaseVAE, device: torch.device) -> Starlette:
    # Lock to ensure only one GPU operation at a time
    gpu_lock = asyncio.Lock()

    async def encode(request: Request) -> Response:
        body = await request.body()
        images = deserialize_tensor(body, device)
        logger.info(f"Encoding images with shape {images.shape}")
        async with gpu_lock:
            with torch.no_grad():
                latents = vae._encode(images)
            result = serialize_tensor(latents)
        return Response(content=result, media_type="application/octet-stream")

    async def decode(request: Request) -> Response:
        body = await request.body()
        latents = deserialize_tensor(body, device)
        logger.info(f"Decoding latents with shape {latents.shape}")
        async with gpu_lock:
            with torch.no_grad():
                images = vae._decode(latents)
            result = serialize_tensor(images)
        return Response(content=result, media_type="application/octet-stream")

    async def health(request: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "pretrained_model_id": vae.pretrained_model_id,
                "revision": vae.revision,
                "subfolder": vae.subfolder,
            }
        )

    routes = [
        Route("/encode", encode, methods=["POST"]),
        Route("/decode", decode, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ]

    return Starlette(routes=routes)


def main():
    config_file = sys.argv[1]
    config = VAEServerConfig(**load_config_file(config_file))

    logger.info(f"Loading VAE model: {config.vae.pretrained_model_id}")
    config.vae.load_model(config.device)

    logger.info(f"Starting VAE server on {config.host}:{config.port}")
    app = create_app(config.vae, config.device)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
