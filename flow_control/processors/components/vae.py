from typing import Annotated, Any, Literal, cast

import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderKLFlux2,
    AutoencoderKLQwenImage,
    ModelMixin,
)
from diffusers.models.autoencoders.autoencoder_kl import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from einops import rearrange
from pydantic import Discriminator, Tag

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.remote import RemoteOffloadable
from flow_control.utils.types import TorchDType

logger = get_logger(__name__)


class BaseVAE[T: ModelMixin](RemoteOffloadable, HfModelLoader[T]):
    endpoint: str | None = None

    @property
    def in_channels(self) -> int:
        return self.model.config["in_channels"]

    def load_model(self, device: torch.device, frozen: bool = True):
        if self.endpoint is not None:
            logger.info(f"Using remote VAE endpoint: {self.endpoint}")
            self._init_remote(device)
            self._model = None
            # Cast to T so don't have to force downstream checks for model usage
            return cast(T, None)
        else:
            res = super().load_model(device, frozen)
            logger.info(
                f"{self.__class__.__name__} requires {self.in_channels} input channels"
            )
            return res

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode(images).latent_dist.sample()

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents).sample

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.is_remote:
            return self._remote_tensor_call("/encode", images)
        return self._encode(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.is_remote:
            return self._remote_tensor_call("/decode", latents)
        return self._decode(latents)

    async def async_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Async version of ``encode`` for use in async pipelines."""
        if self.is_remote:
            return await self._async_remote_tensor_call("/encode", images)
        return self._encode(images)

    async def async_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Async version of ``decode`` for use in async pipelines."""
        if self.is_remote:
            return await self._async_remote_tensor_call("/decode", latents)
        return self._decode(latents)


class Flux1VAE(BaseVAE[AutoencoderKL]):
    type: Literal["flux1"] = "flux1"

    library: Literal["diffusers", "transformers"] = "diffusers"
    class_name: str = "AutoencoderKL"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "vae"
    dtype: TorchDType = torch.bfloat16

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        images = images * 2 - 1
        images = images.to(self.dtype)
        latent = cast(
            AutoencoderKLOutput, self.model.encode(images)
        ).latent_dist.sample()
        latent = (latent - self.model.config["shift_factor"]) * self.model.config[
            "scaling_factor"
        ]
        return latent

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents / self.model.config["scaling_factor"]) + self.model.config[
            "shift_factor"
        ]
        latents = latents.to(self.dtype)
        image = cast(
            DecoderOutput, self.model.decode(cast(torch.FloatTensor, latents))
        ).sample
        image = (image + 1) / 2
        return image


class QwenImageVAE(BaseVAE[AutoencoderKLQwenImage]):
    type: Literal["qwen"] = "qwen"

    library: Literal["diffusers", "transformers"] = "diffusers"
    class_name: str = "AutoencoderKLQwenImage"
    pretrained_model_id: str = "Qwen/Qwen-Image"
    subfolder: str | None = "vae"
    dtype: TorchDType = torch.bfloat16

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        has_frame_dim = images.ndim == 5  # b, c, f, h, w
        if not has_frame_dim:
            images = rearrange(images, "b c h w -> b c 1 h w")
        images = images * 2 - 1
        images = images.to(self.dtype)
        latents = cast(
            AutoencoderKLOutput, self.model.encode(images)
        ).latent_dist.mode()
        latents_mean = (
            torch.tensor(self.model.config["latents_mean"])
            .view(1, self.model.config["z_dim"], 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.model.config["latents_std"])
            .view(1, self.model.config["z_dim"], 1, 1, 1)
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
            torch.tensor(self.model.config["latents_mean"])
            .view(1, self.model.config["z_dim"], 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        # In the original implementation, latents are scaled by 1/stddev during encoding
        latents_std = (
            torch.tensor(self.model.config["latents_std"])
            .view(1, self.model.config["z_dim"], 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        latents = latents.to(self.dtype)
        images = cast(DecoderOutput, self.model.decode(latents)).sample
        images = (images + 1) / 2
        if not has_frame_dim:
            images = rearrange(images, "b c 1 h w -> b c h w")
        return images


class Flux2VAE(BaseVAE[AutoencoderKLFlux2]):
    type: Literal["flux2"] = "flux2"

    library: Literal["diffusers", "transformers"] = "diffusers"
    class_name: str = "AutoencoderKLFlux2"
    pretrained_model_id: str = "black-forest-labs/FLUX.2-dev"
    subfolder: str | None = "vae"
    dtype: TorchDType = torch.bfloat16

    patch_size: int = 2

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def _unpack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def _encode(self, images: torch.Tensor):
        images = images * 2 - 1
        images = images.to(self.dtype)
        latents = cast(
            AutoencoderKLOutput, self.model.encode(images)
        ).latent_dist.sample()

        bn: Any = self.model.bn
        latents_bn_mean = bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            bn.running_var.view(1, -1, 1, 1) + self.model.config["batch_norm_eps"]
        ).to(latents.device, latents.dtype)

        # bn is applied to the 2x2 packed latents
        latents = self._pack_latents(latents)
        latents = (latents - latents_bn_mean) / latents_bn_std
        latents = self._unpack_latents(latents)
        return latents

    def _decode(self, latents):
        latents = latents.to(self.dtype)
        bn: Any = self.model.bn
        latents_bn_mean = bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            bn.running_var.view(1, -1, 1, 1) + self.model.config["batch_norm_eps"]
        ).to(latents.device, latents.dtype)

        latents = self._pack_latents(latents)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpack_latents(latents)

        image = cast(
            DecoderOutput, self.model.decode(cast(torch.FloatTensor, latents))
        ).sample
        image = (image + 1) / 2
        return image


VAE = Annotated[
    Annotated[Flux1VAE, Tag("flux1")]
    | Annotated[QwenImageVAE, Tag("qwen")]
    | Annotated[Flux2VAE, Tag("flux2")],
    Discriminator("type"),
]
