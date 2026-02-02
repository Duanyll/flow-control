from abc import ABC, abstractmethod
from typing import Any, Literal, NotRequired, TypedDict

import torch
from einops import rearrange, repeat
from pydantic import BaseModel, ConfigDict

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.resize import (
    ResolutionList,
    resize_to_closest_resolution,
    resize_to_multiple_of,
)
from flow_control.utils.types import TorchDevice

from .components.encoder import Encoder
from .components.llm import LLMClient
from .components.vae import VAE


class InputBatch(TypedDict):
    pass


class TrainInputBatch(InputBatch):
    pass


class ProcessedBatch(TypedDict):
    image_size: tuple[int, int]
    """Height and width of the images in the batch in pixels. Used for initializing latents."""
    latent_length: NotRequired[int]
    """Length of the latents in the batch. Used for bucket samplers."""
    noisy_latents: NotRequired[torch.Tensor]
    """Noisy latents input to the model."""
    clean_latents: NotRequired[torch.Tensor]
    """Clean latents corresponding to the images in the batch, as training targets."""

    negative: NotRequired[Any]


class BaseProcessor(BaseModel, ABC):
    task: str | None = None
    preset: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---------------------------------- Loading --------------------------------- #

    _encoding_components = [
        "vae",
        "text_encoder",
        "tokenizer",
    ]
    _decoding_components = ["vae"]

    vae: VAE
    encoder: Encoder
    pooled_encoder: Encoder | None = None
    llm: LLMClient | None = None

    _encoding_components: list[str] = []
    _decoding_components: list[str] = []
    device: TorchDevice = torch.device("cuda")

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
            if model_loader is not None:
                model_loader.load_model(device=device)

    # --------------------------- Processing Interfaces -------------------------- #

    @abstractmethod
    async def prepare_inference_batch(self, batch: InputBatch) -> ProcessedBatch:
        """
        Prepares the input batch for inference.
        Should return a ProcessedBatch with all necessary fields.
        """
        raise NotImplementedError()

    def get_negative_batch(self, batch: ProcessedBatch) -> ProcessedBatch | None:
        """
        Retrieves the negative batch from the processed batch if it exists.

        Returns:
            A ProcessedBatch representing the negative batch, or None if not present.
        """
        negative: Any = batch.get("negative", None)
        if negative is not None:
            batch = batch.copy()
            batch.pop("negative")
            batch.update(negative)
            return batch
        else:
            return None

    @abstractmethod
    async def prepare_training_batch(self, batch: TrainInputBatch) -> ProcessedBatch:
        """
        Prepares the input batch for training.
        Should return a ProcessedBatch with all necessary fields.
        """
        raise NotImplementedError()

    def decode_output(
        self, output_latent: torch.Tensor, batch: ProcessedBatch
    ) -> torch.Tensor:
        """
        Decodes the output latents from the model into images.

        Should return a primary image tensor of shape (B, C, H, W), and optionally
        save extra data into the batch if needed.
        """
        return self.decode_latents(output_latent, batch["image_size"])

    def get_latent_length(self, batch: ProcessedBatch) -> int:
        """
        Computes the latent length for the given batch based on its image size. The
        result has not to be the actual length of the transfomer's input, but should be
        useful for bucketing batches of similar sizes (computationally) together.

        Returns:
            The latent length as an integer.
        """
        h, w = batch["image_size"]
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        latent_length = (h * w) // ratio
        return latent_length

    # ----------------------------- Latent Utilities ----------------------------- #

    vae_scale_factor: int = 8
    patch_size: int = 2
    latent_channels: int = 16

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
        batch: ProcessedBatch,
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
        h, w = batch["image_size"]  # type: ignore
        c = self.latent_channels
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        latents = torch.randn(
            (1, c, h, w), generator=generator, device=device, dtype=dtype
        )
        batch["noisy_latents"] = self._pack_latents(latents)
        return batch["noisy_latents"]

    @torch.no_grad()
    def encode_latents(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = repeat(image, "b h w -> b c h w", c=3)
        latents = self.vae.encode(image)
        latents = self._pack_latents(latents)
        return latents

    @torch.no_grad()
    def decode_latents(
        self, latents: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        latents = self._unpack_latents(latents, size)
        image = self.vae.decode(latents)
        return image

    # ---------------------------- Resizing Utilities ---------------------------- #

    resize_mode: Literal["list", "multiple_of"]
    preferred_resolutions: ResolutionList
    default_resolution: tuple[int, int]
    multiple_of: int
    total_pixels: int = 0

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.resize_mode == "list":
            image = resize_to_closest_resolution(image, self.preferred_resolutions)
        elif self.resize_mode == "multiple_of":
            image = resize_to_multiple_of(
                image, multiple=self.multiple_of, pixels=self.total_pixels
            )
        return image

    # ----------------------------- Prompt Utilities ----------------------------- #

    async def chat_completion(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        if self.llm is not None:
            msg, _ = await self.llm.generate(prompt, images, system_prompt)
            return msg
        elif hasattr(self.encoder, "generate"):
            return self.encoder.generate(prompt, images, system_prompt)  # type: ignore
        else:
            raise NotImplementedError("Cannot generate chat completion. Use a ")

    class _EncodePromptReturn(TypedDict):
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: torch.Tensor | None

    def encode_prompt(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> _EncodePromptReturn:
        prompt_embeds = self.encoder.encode(
            prompt, images=images, system_prompt=system_prompt
        )
        pooled_prompt_embeds = (
            self.pooled_encoder.encode(
                prompt, images=images, system_prompt=system_prompt
            )
            if self.pooled_encoder is not None
            else None
        )
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
