import io
import pickle
from typing import Annotated, Literal

import httpx
import torch
from einops import rearrange
from pydantic import PlainValidator

from .hf_model import HfModelLoader
from .logging import get_logger
from .types import TorchDType

logger = get_logger(__name__)

# Use a long timeout for large tensor transfers
REMOTE_VAE_TIMEOUT = httpx.Timeout(timeout=300.0)


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes, converting to bf16 first."""
    tensor = tensor.to(torch.bfloat16).cpu()
    buffer = io.BytesIO()
    pickle.dump(tensor, buffer)
    return buffer.getvalue()


def _deserialize_tensor(
    data: bytes, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Deserialize tensor from bytes."""
    buffer = io.BytesIO(data)
    tensor: torch.Tensor = pickle.load(buffer)  # noqa: S301
    return tensor.to(device=device, dtype=dtype)


class BaseVAE(HfModelLoader):
    endpoint: str | None = None

    def _load_model(self, use_meta_device: bool = False):
        return super().load_model(use_meta_device)

    def load_model(self, use_meta_device=False):
        if self.endpoint is not None:
            logger.info(f"Using remote VAE endpoint: {self.endpoint}")
            self._verify_remote_vae()
            self._model = None
            return None
        else:
            return super().load_model(use_meta_device)

    def _verify_remote_vae(self):
        """Verify that the remote VAE server is available and serves the correct model."""
        with httpx.Client(timeout=REMOTE_VAE_TIMEOUT) as client:
            try:
                response = client.get(f"{self.endpoint}/health")
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise ConnectionError(
                    f"Failed to connect to remote VAE server at {self.endpoint}: {e}"
                ) from e

            info = response.json()
            remote_model_id = info.get("pretrained_model_id")
            remote_revision = info.get("revision")
            remote_subfolder = info.get("subfolder")

            mismatches = []
            if remote_model_id != self.pretrained_model_id:
                mismatches.append(
                    f"pretrained_model_id: local={self.pretrained_model_id}, "
                    f"remote={remote_model_id}"
                )
            if remote_revision != self.revision:
                mismatches.append(
                    f"revision: local={self.revision}, remote={remote_revision}"
                )
            if remote_subfolder != self.subfolder:
                mismatches.append(
                    f"subfolder: local={self.subfolder}, remote={remote_subfolder}"
                )

            if mismatches:
                raise ValueError(
                    "Remote VAE server model mismatch:\n" + "\n".join(mismatches)
                )

            logger.info(
                f"Remote VAE server verified: {remote_model_id} "
                f"(revision={remote_revision}, subfolder={remote_subfolder})"
            )

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode(images).latent_dist.sample()

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents).sample

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.endpoint is not None:
            return self._remote_encode(images)
        else:
            return self._encode(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.endpoint is not None:
            return self._remote_decode(latents)
        else:
            return self._decode(latents)

    def _remote_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using remote VAE server."""
        device = images.device
        dtype = images.dtype
        data = _serialize_tensor(images)
        logger.debug(
            f"Sending {len(data) / 1024 / 1024:.2f} MB to remote VAE for encoding"
        )
        with httpx.Client(timeout=REMOTE_VAE_TIMEOUT) as client:
            response = client.post(f"{self.endpoint}/encode", content=data)
            response.raise_for_status()
        return _deserialize_tensor(response.content, device, dtype)

    def _remote_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents using remote VAE server."""
        device = latents.device
        dtype = latents.dtype
        data = _serialize_tensor(latents)
        logger.debug(
            f"Sending {len(data) / 1024 / 1024:.2f} MB to remote VAE for decoding"
        )
        with httpx.Client(timeout=REMOTE_VAE_TIMEOUT) as client:
            response = client.post(f"{self.endpoint}/decode", content=data)
            response.raise_for_status()
        return _deserialize_tensor(response.content, device, dtype)


class Flux1VAE(BaseVAE):
    library: Literal["diffusers", "transformers"] = "diffusers"
    class_name: str = "AutoencoderKL"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "vae"
    dtype: TorchDType | Literal["auto"] = torch.bfloat16

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        images = images * 2 - 1
        images = images.to(torch.bfloat16)
        latent = self.model.encode(images).latent_dist.sample()
        latent = (
            latent - self.model.config.shift_factor
        ) * self.model.config.scaling_factor
        return latent

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (
            latents / self.model.config.scaling_factor
        ) + self.model.config.shift_factor
        image = self.model.decode(latents).sample
        image = (image + 1) / 2
        return image


class QwenImageVAE(BaseVAE):
    library: Literal["diffusers", "transformers"] = "diffusers"
    class_name: str = "AutoencoderKLQwenImage"
    pretrained_model_id: str = "Qwen/Qwen-Image"
    subfolder: str | None = "vae"
    dtype: TorchDType | Literal["auto"] = torch.bfloat16

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        has_frame_dim = images.ndim == 5  # b, c, f, h, w
        if not has_frame_dim:
            images = rearrange(images, "b c h w -> b c 1 h w")
        images = images * 2 - 1
        images = images.to(torch.bfloat16)
        latents = self.model.encode(images).latent_dist.sample()
        latents_mean = (
            torch.tensor(self.model.config.latents_mean)
            .view(1, self.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.model.config.latents_std)
            .view(1, self.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) / latents_std
        if not has_frame_dim:
            latents = rearrange(latents, "b c 1 h w -> b c h w")
        return latents

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        has_frame_dim = latents.ndim == 5  # b, c, f, h, w
        if not has_frame_dim:
            latents = rearrange(latents, "b c h w -> b c 1 h w")
        latents_mean = (
            torch.tensor(self.model.config.latents_mean)
            .view(1, self.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        # In the original implementation, latents are scaled by 1/stddev during encoding
        latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(
            1, self.model.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        images = self.model.decode(latents).sample
        images = (images + 1) / 2
        if not has_frame_dim:
            images = rearrange(images, "b c 1 h w -> b c h w")
        return images


VAE_REGISTRY = {
    "flux1": Flux1VAE,
    "qwen": QwenImageVAE,
}


def parse_vae(conf: dict) -> BaseVAE:
    vae_type = conf.pop("type")
    vae_class = VAE_REGISTRY.get(vae_type)
    if vae_class is None:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return vae_class(**conf)


VAE = Annotated[BaseVAE, PlainValidator(parse_vae)]
