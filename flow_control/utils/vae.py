from typing import Literal

import torch
from einops import rearrange

from .hf_model import HfModelLoader
from .logging import get_logger
from .types import TorchDType

logger = get_logger(__name__)

class VAE(HfModelLoader):
    endpoint: str | None = None

    def _load_model(self, use_meta_device: bool = False):
        return super().load_model(use_meta_device)

    def load_model(self, use_meta_device = False):
        if self.endpoint is not None:
            logger.info(f"Skipping VAE loading, using remote endpoint: {self.endpoint}")
            self._model = None
            return None
        else:
            return super().load_model(use_meta_device)
        
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode(images).latent_dist.sample()
    
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents).sample
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.endpoint is not None:
            # TODO: Implement remote VAE encoding
            raise NotImplementedError("Remote VAE encoding is not implemented.")
        else:
            return self._encode(images)
        
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.endpoint is not None:
            # TODO: Implement remote VAE decoding
            raise NotImplementedError("Remote VAE decoding is not implemented.")
        else:
            return self._decode(latents)
        
    
class Flux1VAE(VAE):
    type: Literal["diffusers", "transformers"] = "diffusers"
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
    

class QwenImageVAE(VAE):
    type: Literal["diffusers", "transformers"] = "diffusers"
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
        latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(
            1, self.model.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std
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
        latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(
            1, self.model.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean
        images = self.model.decode(latents).sample
        images = (images + 1) / 2
        if not has_frame_dim:
            images = rearrange(images, "b c 1 h w -> b c h w")
        return images