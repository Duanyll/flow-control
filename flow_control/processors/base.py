import io
from abc import ABC, abstractmethod
from typing import Literal, NotRequired, TypedDict

import requests
import torch
from einops import rearrange
from PIL import Image
from pydantic import BaseModel, ConfigDict

from flow_control.utils.common import tensor_to_pil
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

    _loading_preset: dict[
        str, list[Literal["encode", "decode", "preview", "allow_remote_preview"]]
    ] = {}
    device: TorchDevice = torch.device("cuda")

    vae_scale_factor: int = 8
    patch_size: int = 2
    latent_channels: int = 16
    default_resolution: tuple[int, int] = (1024, 1024)

    def load_models(
        self,
        preset: list[Literal["encode", "decode", "preview"]],
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = self.device
        for field_name, field_preset in self._loading_preset.items():
            if any(p in field_preset for p in preset):
                if (
                    preset == ["preview"]
                    and "allow_remote_preview" in field_preset
                    and getattr(self, field_name).endpoint is not None
                ):
                    setattr(self, f"_{field_name}", None)
                    continue

                model = getattr(self, field_name).load_model()
                if hasattr(model, "to"):
                    model.to(device)
                setattr(self, f"_{field_name}", model)
            else:
                setattr(self, f"_{field_name}", None)

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

    def preview_output(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> Image.Image:
        """
        Decodes the output latents from the model into a preview image. It does not need
        to be as high-quality as `decode_output`, but should be efficient to be called
        frequently during training. It may call remote endpoints to save local VRAM if
        configured to do so.

        Should return a PIL Image object representing the preview.
        """
        if hasattr(self, "vae") and self.vae.endpoint is not None:  # type: ignore
            return self._remote_decode_outputs(output_latent, batch)
        else:
            return tensor_to_pil(self.decode_output(output_latent, batch))

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
        self, batch: BatchType, generator: torch.Generator | None = None, device=None
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
        latents = torch.randn((1, c, h, w), generator=generator, device=device)
        batch["noisy_latents"] = self._pack_latents(latents)
        return batch["noisy_latents"]

    def _remote_decode_outputs(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> Image.Image:
        """
        Decodes the output latents using a remote endpoint configured in the VAE.

        Should return a PIL Image object representing the decoded image.
        """
        batch["noisy_latents"] = output_latent
        buffer = io.BytesIO()
        torch.save(batch, buffer)
        buffer.seek(0)

        response = requests.post(
            self.vae.endpoint,  # type: ignore
            files={"file": ("batch.pt", buffer, "application/octet-stream")},
            verify=False,
        )

        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(
                f"Remote decoding failed with status {response.status_code}: {error_detail}"
            )

        image_data = io.BytesIO(response.content)
        pil_image = Image.open(image_data)

        return pil_image
