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


class BaseVAE[T: ModelMixin](HfModelLoader[T]):
    """Base class for VAE encode / decode.

    Subclasses override ``_encode`` to apply model-specific normalization.
    The ``posterior`` parameter selects the posterior handling strategy; see
    :data:`PosteriorMode`.  The default implementation always returns the mode
    (suitable for VAEs with negligible variance).
    """

    @property
    def in_channels(self) -> int:
        return self.model.config["in_channels"]

    def load_model(self, device: torch.device, frozen: bool = True) -> bool:
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
        return self._encode(images, posterior=posterior)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
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


@vae_registry.register("qwen21")
class QwenImage21VAE(QwenImageVAE):
    """Qwen21 keeps Qwen's latent normalization, with 64 channels and 16x compression."""

    type: Literal["qwen21"] = "qwen21"
    class_name: str = "AutoencoderKLQwenImage21"
    pretrained_model_id: str = "Qwen/Qwen-Image-2.1"

    @property
    def in_channels(self) -> int:
        return self.model.config["in_channels"]


@vae_registry.register("cosmos3")
class Cosmos3VAE(QwenImageVAE):
    """Cosmos3 ships the Wan2.2-TI2V VAE: 16x spatial, 4x temporal, 48 channels.

    Its latent normalization is per-channel mean/std from the config, exactly
    like Qwen-Image's, so only the loader coordinates differ. Images go through
    as single-frame video, which is what ``num_frames=1`` means upstream.
    """

    type: Literal["cosmos3"] = "cosmos3"
    class_name: str = "AutoencoderKLWan"
    pretrained_model_id: str = "nvidia/Cosmos3-Nano"
    subfolder: str | None = "vae"

    @property
    def in_channels(self) -> int:
        # The Wan VAE pixel-shuffles 2x2 before its first conv, so `in_channels`
        # in the config is 12; callers care about the 3 RGB channels they pass.
        return 3


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

    def _stats(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel ``(mean, std)`` of the packed latents; the normalization runs in their dtype."""
        bn: Any = self.model.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
        std = torch.sqrt(
            bn.running_var.view(1, -1, 1, 1) + self.model.config["batch_norm_eps"]
        ).to(device, dtype)
        return mean, std

    def _normalize(self, latents: torch.Tensor, shift: bool) -> torch.Tensor:
        mean, std = self._stats(latents.device, self.dtype)
        packed = self._pack_latents(latents).to(mean.dtype)
        return self._unpack_latents((packed - mean if shift else packed) / std)

    def _encode(
        self, images: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        """Encode and normalize per packed channel.

        Unlike Flux1/Qwen, Flux2's posterior variance is significant, so all
        three posterior modes are implemented here. The normalization is linear,
        so ``"distribution"`` applies it to the mean and the std separately and
        returns ``[2*B, C, H, W]`` with means and stds stacked on dim 0.
        """
        images = (images * 2 - 1).to(self.dtype)
        latent_dist = cast(AutoencoderKLOutput, self.model.encode(images)).latent_dist
        if posterior == "distribution":
            return torch.cat(
                [
                    self._normalize(latent_dist.mean, shift=True),
                    self._normalize(latent_dist.std, shift=False),
                ],
                dim=0,
            )
        latents = latent_dist.mode() if posterior == "mode" else latent_dist.sample()
        return self._normalize(latents, shift=True)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(latents.device, self.dtype)
        packed = self._pack_latents(latents).to(mean.dtype) * std + mean
        latents = self._unpack_latents(packed).to(self.dtype)
        image = cast(
            DecoderOutput, self.model.decode(cast(torch.FloatTensor, latents))
        ).sample
        return (image + 1) / 2


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


# Ideogram 4's latent normalization (official ``latent_norm.py``), per packed channel in
# ``(ph pw c)`` order. The checkpoint's ``bn`` stats are FLUX.2's and differ from these.
IDEOGRAM4_LATENT_SHIFT = (
    0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549,
    0.05021099, -0.15083604, -0.15360136, -0.20131799, 0.01922352, 0.06226260,
    0.10140969, -0.06739428, 0.37582610, -0.23371200, 0.35164491, -0.02590912,
    -0.02719350, -0.10833897, -0.14768480, -0.01130957, -0.22983720, 0.23526423,
    -0.10893522, 0.11957631, 0.04047799, 0.31345890, -0.17225064, -0.18646109,
    -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294, 0.26952152,
    -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.15485020, -0.18971131,
    0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937,
    0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137,
    -0.21863696, 0.23228983, -0.11739769, 0.11693044, 0.02563311, 0.31356594,
    -0.17420591, -0.19006285, -0.34905377, -0.04025005, 0.01924137, 0.07652984,
    0.29956080, 0.26280570, -0.22011674, -0.12715361, 0.04879879, -0.14075719,
    -0.15935895, -0.21235840, 0.01974813, 0.05523547, 0.10011992, -0.06428964,
    0.37781868, -0.21491644, 0.34254215, -0.03153528, -0.03100820, -0.10761415,
    -0.14730405, -0.02475182, -0.22855880, 0.25150810, -0.10445128, 0.12446000,
    0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.01291770,
    0.02578168, 0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155,
    0.04059549, -0.14043529, -0.16111187, -0.20203050, 0.02602069, 0.04852717,
    0.10432153, -0.06309942, 0.38402443, -0.22397003, 0.34814481, -0.03774432,
    -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016, 0.24872463,
    -0.11399775, 0.12226870, 0.05620835, 0.30917800, -0.18065738, -0.19401479,
    -0.34495114, -0.01760592,
)  # fmt: skip
IDEOGRAM4_LATENT_SCALE = (
    1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.66753160, 1.69059584,
    1.56853198, 1.62314944, 1.89106626, 1.58086668, 1.60822129, 1.60962993,
    1.63322129, 1.56074359, 1.73419528, 1.79192650, 1.64040632, 1.66802808,
    1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046,
    1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641,
    1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756, 1.91310663,
    1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539,
    1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606,
    1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667,
    1.61659342, 1.59722044, 1.64103121, 1.54085310, 1.68610394, 1.67772755,
    1.78998563, 1.66621713, 1.65458955, 1.66041308, 1.64710857, 1.68163503,
    1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
    1.87618195, 1.55843750, 1.57438785, 1.61711053, 1.63094305, 1.55644029,
    1.73124302, 1.80666627, 1.64636210, 1.65932006, 1.60816188, 1.75682671,
    1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068,
    1.65534289, 1.67132281, 1.80317197, 1.67673140, 1.65700938, 1.68426259,
    1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117,
    1.57309060, 1.61928553, 1.87148809, 1.56244866, 1.56697152, 1.61584394,
    1.62759496, 1.55480378, 1.73484107, 1.79055143, 1.64688773, 1.66121492,
    1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592, 1.60792883,
    1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
    1.65655173, 1.68533454,
)  # fmt: skip


@vae_registry.register("ideogram4")
class Ideogram4VAE(Flux2VAE):
    """Ideogram 4's FLUX.2 autoencoder with the official latent normalization: fp32
    per-channel shift and scale over the ``(ph pw c)``-packed latents (the model's
    channel order), not the checkpoint's ``bn`` stats in ``(c ph pw)`` order."""

    type: Literal["ideogram4"] = "ideogram4"
    pretrained_model_id: str = "ideogram-ai/ideogram-4-fp8"

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "b c (h p1) (w p2) -> b (p1 p2 c) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def _unpack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "b (p1 p2 c) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def _stats(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The official constants are applied in fp32 whatever the model dtype.
        shift = torch.tensor(IDEOGRAM4_LATENT_SHIFT, device=device).view(1, -1, 1, 1)
        scale = torch.tensor(IDEOGRAM4_LATENT_SCALE, device=device).view(1, -1, 1, 1)
        return shift, scale


VAE = Annotated[BaseVAE, RegistryUnion(vae_registry, "type")]
