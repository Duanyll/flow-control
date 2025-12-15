from abc import ABC, abstractmethod
from typing import Literal, NotRequired, TypedDict

import torch
from einops import rearrange
from pydantic import BaseModel, ConfigDict

from flow_control.utils.types import TorchDevice


class BaseProcessor(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class BatchType(TypedDict):
        image_size: NotRequired[tuple[int, int]]

    _loading_preset: dict[str, list[Literal["encode", "decode", "always"]]] = {}
    device: TorchDevice = torch.device("cuda")

    vae_scale_factor: int = 8
    patch_size: int = 2
    latent_channels: int = 16
    default_resolution: tuple[int, int] = (1024, 1024)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for field_name, preset in self._loading_preset.items():
            if "always" in preset:
                model = getattr(self, field_name).load_model()
                setattr(self, f"_{field_name}", model)

    def load_models(
        self,
        preset: list[Literal["encode", "decode"]],
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = self.device
        for field_name, field_preset in self._loading_preset.items():
            if any(p in field_preset for p in preset):
                model = getattr(self, field_name).load_model()
                model.to(device)
                setattr(self, f"_{field_name}", model)
            elif "always" not in field_preset:
                setattr(self, f"_{field_name}", None)

    @abstractmethod
    def preprocess_batch(self, batch: BatchType) -> BatchType:
        raise NotImplementedError()

    @abstractmethod
    def make_negative_batch(self, batch: BatchType) -> BatchType:
        raise NotImplementedError()

    @abstractmethod
    def decode_output(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> torch.Tensor:
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
        self, batch: BatchType, generator: torch.Generator | None = None, device=None
    ) -> torch.Tensor:
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
        return self._pack_latents(latents)
