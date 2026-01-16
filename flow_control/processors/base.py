from abc import ABC, abstractmethod
from typing import Literal, NotRequired, TypedDict

import torch
from einops import rearrange
from pydantic import BaseModel, ConfigDict

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.types import TorchDevice


class BaseProcessor(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class BatchType(TypedDict):
        image_size: NotRequired[tuple[int, int]]
        """Height and width of the images in the batch in pixels. Used for initializing latents."""
        latent_length: NotRequired[int]
        """Length of the latents in the batch. Used for bucket samplers."""
        noisy_latents: NotRequired[torch.Tensor]
        """Noisy latents input to the model."""
        clean_latents: NotRequired[torch.Tensor]
        """Clean latents corresponding to the images in the batch, as training targets."""

    _encoding_components: list[str] = []
    _decoding_components: list[str] = []
    device: TorchDevice = torch.device("cuda")

    vae_scale_factor: int = 8
    patch_size: int = 2
    latent_channels: int = 16
    default_resolution: tuple[int, int] = (1024, 1024)

    def load_models(
        self,
        mode: Literal["encode", "decode"],
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = self.device
        self.device = device
        field_list = (
            self._encoding_components if mode == "encode" else self._decoding_components
        )
        for field_name in field_list:
            model_loader: HfModelLoader = getattr(self, field_name)
            model_loader.load_model()
            if model_loader.model is not None and hasattr(model_loader.model, "to"):
                model_loader.model.to(device)

    @abstractmethod
    def preprocess_batch(self, batch: BatchType) -> BatchType:
        """
        Preprocesses the input batch before feeding it to the model.

        Should modify the batch in-place and return it.
        """
        raise NotImplementedError()

    @abstractmethod
    def make_negative_batch(self, batch: BatchType) -> BatchType:
        """
        Transforms the input batch into a negative batch for CFG. Should accept both
        preprocessed and unprocessed batches, and return a preprocessed negative batch
        that is ready to be fed to the model.

        Should modify the batch in-place and return it. It's caller's responsibility to
        clone the input batch if needed.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_output(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> torch.Tensor:
        """
        Decodes the output latents from the model into images.

        Should return a primary image tensor of shape (B, C, H, W), and optionally
        save extra data into the batch if needed.
        """
        raise NotImplementedError()

    def _pack_latents(self, latents):
        return rearrange(
            latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def _unpack_latents(self, latents, size: tuple[int, int]):
        h, w = size
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        return rearrange(
            latents,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h // self.patch_size,
            w=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def initialize_latents(
        self,
        batch: BatchType,
        generator: torch.Generator | None = None,
        device=None,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        """
        Initializes noisy latents for the given batch based on its image size.

        Modifies the batch in-place to add the "noisy_latents" key and returns the
        initialized latents.
        """
        if device is None:
            device = self.device
        if "image_size" in batch:
            h, w = batch["image_size"]
        else:
            h, w = self.default_resolution
        c = self.latent_channels
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        latents = torch.randn(
            (1, c, h, w), generator=generator, device=device, dtype=dtype
        )
        batch["noisy_latents"] = self._pack_latents(latents)
        return batch["noisy_latents"]
