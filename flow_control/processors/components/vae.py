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

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.registry import Registry, RegistryUnion
from flow_control.utils.remote import RemoteOffloadable
from flow_control.utils.types import TorchDType

logger = get_logger(__name__)

PosteriorMode = Literal["mode", "sample", "distribution"]
"""How to handle the VAE posterior distribution during encoding.

- ``"mode"``: Return the distribution mean (deterministic).
- ``"sample"``: Draw a single sample from the posterior.
- ``"distribution"``: Return mean **and** std stacked along the batch dimension
  (``[2*B, C, H, W]``), deferring the sampling to training time so that
  each epoch sees a different latent for the same image.

For Flux1 and Qwen VAEs the posterior variance is negligible in bf16, so all
three modes fall back to ``"mode"`` internally.  Only Flux2VAE, whose VAE was
trained differently and has significant variance, implements all three modes.
"""


class BaseVAE[T: ModelMixin](RemoteOffloadable, HfModelLoader[T]):
    """Base class for VAE encode / decode.

    Subclasses override ``_encode`` to apply model-specific normalization.
    The ``posterior`` parameter selects the posterior handling strategy; see
    :data:`PosteriorMode`.  The default implementation always returns the mode
    (suitable for VAEs with negligible variance).
    """

    endpoint: str | None = None

    @property
    def in_channels(self) -> int:
        return self.model.config["in_channels"]

    def load_model(self, device: torch.device, frozen: bool = True) -> bool:
        if self.endpoint is not None:
            logger.info(f"Using remote VAE endpoint: {self.endpoint}")
            self._init_remote(device)
            self._model = None
            return False
        fresh = super().load_model(device, frozen)
        logger.info(
            f"{self.__class__.__name__} requires {self.in_channels} input channels"
        )
        return fresh

    def _encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        # Default: always return mode (posterior variance negligible in bf16)
        return self.model.encode(images).latent_dist.mode()

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents).sample

    def encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        if self.is_remote:
            return self._remote_tensor_call(f"/encode?posterior={posterior}", images)
        return self._encode(images, posterior=posterior)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.is_remote:
            return self._remote_tensor_call("/decode", latents)
        return self._decode(latents)

    async def async_encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        """Async version of ``encode`` for use in async pipelines."""
        if self.is_remote:
            return await self._async_remote_tensor_call(
                f"/encode?posterior={posterior}", images
            )
        return self._encode(images, posterior=posterior)

    async def async_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Async version of ``decode`` for use in async pipelines."""
        if self.is_remote:
            return await self._async_remote_tensor_call("/decode", latents)
        return self._decode(latents)


vae_registry: Registry[BaseVAE] = Registry("vae", base=BaseVAE)


@vae_registry.register("flux1")
class Flux1VAE(BaseVAE[AutoencoderKL]):
    type: Literal["flux1"] = "flux1"

    library: Literal["diffusers", "transformers"] = "diffusers"
    class_name: str = "AutoencoderKL"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "vae"
    dtype: TorchDType = torch.bfloat16

    def _encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        images = images * 2 - 1
        images = images.to(self.dtype)
        latent = cast(AutoencoderKLOutput, self.model.encode(images)).latent_dist.mode()
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


@vae_registry.register("qwen")
class QwenImageVAE(BaseVAE[AutoencoderKLQwenImage]):
    type: Literal["qwen"] = "qwen"

    library: Literal["diffusers", "transformers"] = "diffusers"
    class_name: str = "AutoencoderKLQwenImage"
    pretrained_model_id: str = "Qwen/Qwen-Image"
    subfolder: str | None = "vae"
    dtype: TorchDType = torch.bfloat16

    @property
    def in_channels(self) -> int:
        return self.model.config["input_channels"]

    def _encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
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


@vae_registry.register("flux2")
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

    def _encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        """Encode with Flux2-specific batch-norm normalization.

        Unlike Flux1/Qwen, Flux2's posterior variance is significant, so all
        three posterior modes are implemented here.

        For ``"distribution"`` mode, the BN normalization ``f(x) = (x - μ) / σ``
        is linear, so we can apply it to mean and std independently:
        ``normalized_mean = (mean - μ) / σ``, ``normalized_std = std / σ``.
        The result is ``[2*B, C, H, W]`` with means and stds stacked on dim 0.
        """
        images = images * 2 - 1
        images = images.to(self.dtype)
        latent_dist = cast(AutoencoderKLOutput, self.model.encode(images)).latent_dist

        # BN stats live in 2x2-packed channel space
        bn: Any = self.model.bn
        bn_mean = bn.running_mean.view(1, -1, 1, 1).to(images.device, images.dtype)
        bn_std = torch.sqrt(
            bn.running_var.view(1, -1, 1, 1) + self.model.config["batch_norm_eps"]
        ).to(images.device, images.dtype)

        if posterior == "distribution":
            mean = latent_dist.mean
            std = latent_dist.std
            packed_mean = self._pack_latents(mean)
            packed_std = self._pack_latents(std)
            normalized_mean = (packed_mean - bn_mean) / bn_std
            normalized_std = packed_std / bn_std
            return torch.cat(
                [
                    self._unpack_latents(normalized_mean),
                    self._unpack_latents(normalized_std),
                ],
                dim=0,
            )

        latents = latent_dist.mode() if posterior == "mode" else latent_dist.sample()
        latents = self._pack_latents(latents)
        latents = (latents - bn_mean) / bn_std
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


@vae_registry.register("identity")
class IdentityVAE(BaseVAE[ModelMixin]):
    """Pass-through "VAE" for pixel-space models: latents *are* pixels.

    ``_encode`` maps ``[0, 1]`` images to ``(2x - 1) * scaling``. Absorbing the
    model's forward-process noise scale into ``scaling`` turns a scaled-noise
    pixel flow into a standard unit-noise rectified flow in latent space: with
    ``scaling = 1/8``, HiDream-O1's ``x_t = (1-t)*x0 + t*(8*eps)`` becomes
    ``z_t = (1-t)*z0 + t*eps`` for ``z = x/8``, so every unit-noise assumption
    in the framework (``initialize_latents``, trainers' ``randn_like``, SDE
    solvers) holds unchanged. The adapter multiplies by ``1/scaling`` before
    feeding pixels to the model.

    There is no posterior: all :data:`PosteriorMode` values return the same
    deterministic result. Nothing is loaded from disk.
    """

    type: Literal["identity"] = "identity"

    library: Literal["diffusers", "transformers", "custom"] = "transformers"
    class_name: str = ""
    pretrained_model_id: str = ""
    dtype: TorchDType = torch.bfloat16

    scaling: float = 0.125

    @property
    def in_channels(self) -> int:
        return 3

    def load_model(self, device: torch.device, frozen: bool = True) -> bool:
        # Nothing to load; keep _model unset so misuse fails loudly.
        return False

    def _encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        return (images.to(self.dtype) * 2 - 1) * self.scaling

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return ((latents / self.scaling + 1) / 2).clamp(0, 1)


VAE = Annotated[BaseVAE, RegistryUnion(vae_registry, "type")]
