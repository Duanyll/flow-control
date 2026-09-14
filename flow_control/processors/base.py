from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, NotRequired, TypedDict, TypeVar

import torch
from einops import rearrange
from pydantic import BaseModel, ConfigDict, Field

from flow_control.data.coercion import JsonBeforeValidator
from flow_control.data.rows import Row
from flow_control.utils.device import default_device
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.registry import Registry
from flow_control.utils.resize import (
    ResolutionList,
    resize_to_closest_resolution,
    resize_to_multiple_of,
)
from flow_control.utils.tensor import ensure_alpha_channel, remove_alpha_channel
from flow_control.utils.types import TorchDevice

from .components.encoder import Encoder, GenerativeEncoder, T5TextEncoder
from .components.llm import LLMClient
from .components.vae import VAE, Flux1VAE, PosteriorMode


class InputRow(TypedDict):
    # Pydantic reads this out of the class body; ty only expects annotations.
    __pydantic_config__ = ConfigDict(  # ty: ignore[invalid-typed-dict-statement]
        extra="allow", arbitrary_types_allowed=True
    )
    image_size: NotRequired[Annotated[tuple[int, int], JsonBeforeValidator] | None]


class TrainInputRow(TypedDict):
    # Pydantic reads this out of the class body; ty only expects annotations.
    __pydantic_config__ = ConfigDict(  # ty: ignore[invalid-typed-dict-statement]
        extra="allow", arbitrary_types_allowed=True
    )
    image_size: NotRequired[Annotated[tuple[int, int], JsonBeforeValidator] | None]


class ProcessedRow(TypedDict):
    image_size: tuple[int, int]
    """Height and width of the images in the row in pixels. Used for initializing latents."""
    __key__: NotRequired[str]
    """Sample identifier carried over from the source dataset, used to name outputs."""
    cost: NotRequired[int]
    """Token total (latent + text + reference) from ``get_cost``; the plan sort key."""
    noisy_latents: NotRequired[torch.Tensor]
    """Noisy latents input to the model."""
    clean_latents: NotRequired[torch.Tensor]
    """Clean latents corresponding to the images in the row, as training targets."""

    negative: NotRequired[Mapping[str, Any]]
    model_image_size: NotRequired[tuple[int, int]]
    """Image size the model sees in one forward (for example one tile); defaults to
    ``image_size``. Resolution-dependent shifts read this, not ``image_size``."""
    tiling: NotRequired[dict[str, Any]]
    """Serialized ``TileLayout``; model evaluation tiles this image on the token grid."""
    tiles: NotRequired[list["ProcessedRow"]]
    """Complete per-tile conditioning rows in row-major order, when sampling tiled."""


class DecodedRow(TypedDict):
    clean_image: torch.Tensor
    """Primary clean image decoded from the latents."""


def sample_posterior(latents: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """``[2, ...]`` (mean, std) -> one fp32 ``[1, ...]`` draw; other shapes unchanged.

    The draw is made and kept in fp32 on purpose: caches hold the posterior in the
    VAE dtype (bf16), where ``std * eps`` is below the resolution of ``mean`` for
    every VAE except Qwen-Image's, so a bf16 draw (or an fp32 draw cast back to
    bf16) collapses to the mean. Consumers promote targets to fp32 anyway.
    """
    if latents.shape[0] != 2:
        return latents
    mean, std = latents[0:1].float(), latents[1:2].float()
    eps = torch.randn(
        mean.shape, generator=generator, device=mean.device, dtype=torch.float32
    )
    return mean + std * eps


TInput = TypeVar("TInput", bound=InputRow)
TTrainInput = TypeVar("TTrainInput", bound=TrainInputRow)
TProcessed = TypeVar("TProcessed", bound=ProcessedRow)


class BaseProcessor[
    TInput: InputRow,
    TTrainInput: TrainInputRow,
    TProcessed: ProcessedRow,
](BaseModel, ABC):
    task: str
    preset: str = ""
    model_config = ConfigDict(extra="forbid")

    # ---------------------------------- Loading --------------------------------- #

    vae: VAE = Flux1VAE()
    encoder: Encoder = T5TextEncoder()
    pooled_encoder: Encoder | None = None
    llm: LLMClient | None = None

    _encoding_components: list[str] = [
        "vae",
        "encoder",
        "pooled_encoder",
    ]
    _decoding_components: list[str] = ["vae"]
    device: TorchDevice = Field(default_factory=default_device)

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
    async def prepare_inference_row(self, row: TInput) -> ProcessedRow:
        """
        Prepares the input row for inference.
        Should return a ProcessedRow with all necessary fields.
        """
        raise NotImplementedError()

    def get_negative_row(self, row: ProcessedRow) -> ProcessedRow | None:
        """
        Retrieves the negative row from the processed row if it exists.

        Returns:
            A ProcessedRow representing the negative row, or None if not present.
        """
        negative: Any = row.get("negative", None)
        if negative is not None:
            row = row.copy()
            row.pop("negative")
            row.update(negative)
            return row
        else:
            return None

    @abstractmethod
    async def prepare_training_row(self, row: TTrainInput) -> ProcessedRow:
        """
        Prepares the input row for training.
        Should return a ProcessedRow with all necessary fields.
        """
        raise NotImplementedError()

    def decode_output(self, output_latent: torch.Tensor, row: TProcessed) -> DecodedRow:
        """
        Decodes the output latents from the model into images.

        Should return a primary image tensor of shape (B, C, H, W), and optionally
        save extra data into the row if needed.
        """
        return {
            "clean_image": self.decode_latents(output_latent, size=row["image_size"])
        }

    def annotate_output(self, decoded: DecodedRow, row: TProcessed) -> torch.Tensor:
        """Compose ONE labeled preview image from a decoded output row.

        ``decode_output`` returns the clean ``clean_image`` (used for reward scoring
        and the report records) plus any task-specific auxiliary tensors; ``annotate_output``
        merges those into a single image for logging/saving, leaving ``clean_image``
        untouched. The default returns ``clean_image`` unchanged.
        """
        return decoded["clean_image"]

    def get_cost(self, row: TProcessed) -> int:
        """Token total of one forward on ``row``; the plan sort key (design D4).

        The default counts latent tokens from ``image_size``; tasks add their text
        (``prompt_embeds.shape[1]``) and reference tokens. It is a work estimate
        for grouping rows of similar cost, not necessarily the transformer's
        exact input length.
        """
        h, w = row["image_size"]
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        return (h * w) // ratio

    # ----------------------------- Latent Utilities ----------------------------- #

    vae_scale_factor: int = 8
    patch_size: int = 2
    latent_channels: int = 16
    initial_noise_scale: float = 1.0
    """Scale applied to the initial sampling noise in :meth:`initialize_latents`.
    Some models deliberately start below the training noise level (HiDream-O1's
    official pipeline initializes at 7.5/8 = 0.9375 of its noise scale)."""
    target_posterior: PosteriorMode = "distribution"
    condition_posterior: PosteriorMode = "mode"
    posterior_fields: ClassVar[tuple[str, ...]] = ("clean_latents",)
    """Row fields that may hold a cached VAE posterior ``[2, ...]`` (mean, std)
    instead of a sample ``[1, ...]``; ``resample`` draws from them. Tasks extend
    this with every field they encode through the VAE (TIE adds
    ``reference_latents``)."""

    def resample(self, row: Row, generator: torch.Generator) -> Row:
        """Per-fetch randomization of ``row`` (design §8); modifies and returns it.

        Every ``posterior_fields`` tensor whose first dim is 2 is replaced by one
        fp32 draw ``mean + std * eps`` of shape ``[1, ...]`` (see
        ``sample_posterior``); list fields are sampled element-wise. Fields already holding a sample (first dim 1, which is what
        ``target_posterior="mode"`` caches) pass through untouched. Only the
        declared fields are looked at, never a name pattern. ``generator`` must
        live on the row's device. Subclasses stack data augmentation on top.
        """
        for name in self.posterior_fields:
            value = row.get(name)
            if isinstance(value, torch.Tensor):
                row[name] = sample_posterior(value, generator)
            elif isinstance(value, list):
                row[name] = [
                    sample_posterior(v, generator) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
        return row

    def _pack_latents(self, latents) -> torch.Tensor:
        return rearrange(
            latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def _unpack_latents(self, latents, size: tuple[int, int]) -> torch.Tensor:
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
        row: TProcessed,
        generator: torch.Generator | None = None,
        device=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """
        Initializes noisy latents for the given row based on its image size.

        Modifies the row in-place to add the "noisy_latents" key and returns the
        initialized latents.
        """
        if device is None:
            device = self.device
        h, w = row["image_size"]
        c = self.latent_channels
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        latents = torch.randn(
            (1, c, h, w), generator=generator, device=device, dtype=dtype
        )
        if self.initial_noise_scale != 1.0:
            latents = (latents.float() * self.initial_noise_scale).to(dtype)
        noisy_latents = row["noisy_latents"] = self._pack_latents(latents)
        return noisy_latents

    def _adapt_image_channels(self, image: torch.Tensor) -> torch.Tensor:
        """Adapt image channels to match the VAE's expected input channels."""
        expected = self.vae.in_channels
        actual = image.shape[1]
        if actual == expected:
            return image
        if expected == 3 and actual == 4:
            return remove_alpha_channel(image)
        if expected == 4 and actual == 3:
            return ensure_alpha_channel(image)
        return image

    @torch.no_grad()
    def encode_latents(
        self, image: torch.Tensor, posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        image = self._adapt_image_channels(image)
        latents = self.vae.encode(image, posterior=posterior)
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

    resize_mode: Literal["list", "multiple_of"] = "multiple_of"
    preferred_resolutions: ResolutionList = []
    default_resolution: tuple[int, int] = (1024, 1024)
    multiple_of: int = 32
    total_pixels: int = 0
    no_upscale: bool = False

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.resize_mode == "list":
            image = resize_to_closest_resolution(image, self.preferred_resolutions)
        elif self.resize_mode == "multiple_of":
            image = resize_to_multiple_of(
                image,
                multiple=self.multiple_of,
                pixels=self.total_pixels,
                no_upscale=self.no_upscale,
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
        elif isinstance(self.encoder, GenerativeEncoder):
            return self.encoder.generate(prompt, images, system_prompt)
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


task_registry: Registry[BaseProcessor] = Registry("processor_task", base=BaseProcessor)
