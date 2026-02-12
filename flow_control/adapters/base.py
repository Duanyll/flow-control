from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from diffusers import ModelMixin
from einops import rearrange
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.types import TorchDType
from flow_control.utils.upcasting import (
    apply_layerwise_upcasting,
    cast_trainable_parameters,
)

logger = get_logger(__name__)


class Batch(TypedDict):
    image_size: tuple[int, int]
    """`(H, W)` The size of the image to generate."""
    clean_latents: torch.Tensor
    """`[B, N, D]` The clean latents. Only available during training."""
    noisy_latents: torch.Tensor
    """`[B, N, D]` The noisy latents to denoise."""


class BaseModelAdapter[TModel: ModelMixin, TBatch: Batch](BaseModel, ABC):
    """
    Base class for all control adapters.
    """

    arch: str
    type: str | list[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def transformer(self) -> TModel:
        return self.hf_model.model

    @transformer.setter
    def transformer(self, value: TModel) -> None:
        self.hf_model.model = value

    @property
    def device(self) -> torch.device:
        return self.transformer.device

    hf_model: HfModelLoader[TModel]
    storage_dtype: TorchDType | None = None
    trainable_dtype: TorchDType = torch.bfloat16

    all_trainable: bool = False
    peft_lora_config: LoraConfig = LoraConfig()
    peft_lora_rank: int = 0
    extra_trainable_modules: list[str] = []

    patch_size: int = 2
    latent_channels: int = 16

    @property
    def dtype(self) -> torch.dtype:
        # Ensure we are getting the correct dtype even after upcasting
        return (
            self.hf_model.dtype
            if self.hf_model.dtype != "auto"
            else self.transformer.dtype
        )

    def load_transformer(self, device: torch.device) -> None:
        self.transformer = self.hf_model.load_model(device=device)
        self.transformer.requires_grad_(self.all_trainable)

        self._install_modules()

        if self.peft_lora_rank > 0:
            self.peft_lora_config.r = self.peft_lora_rank
            if self.peft_lora_config.target_modules == "all-linear":
                self.peft_lora_config.target_modules = list(
                    {
                        k
                        for k, v in self.transformer.named_modules()
                        if isinstance(v, torch.nn.Linear)
                    }
                )
            self.transformer.add_adapter(self.peft_lora_config)

        for name, param in self.transformer.named_parameters():
            if any(k in name for k in self.extra_trainable_modules):
                param.requires_grad = True

        cast_trainable_parameters(self.transformer, self.trainable_dtype)
        if (
            self.hf_model.dtype != "auto"
            and self.storage_dtype is not None
            and self.storage_dtype != self.hf_model.dtype
        ):
            apply_layerwise_upcasting(
                self.transformer,
                storage_dtype=self.storage_dtype,
                compute_dtype=self.hf_model.dtype,
            )
            logger.info(
                f"Applied layerwise casting with storage dtype {self.storage_dtype} and compute dtype {self.hf_model.dtype}"
            )

    def _install_modules(self):
        """
        Create and initialize additional modules on the base model. Called after base model is
        created, before installing PEFT adapters and upcasting.
        """
        pass

    def filter_state_dict(self, state_dict: dict) -> dict:
        """
        Filter the state_dict before saving. By default, only the trainable parameters are kept.

        :param state_dict: The original state_dict.
        :return: The filtered state_dict.
        """
        return {
            k: state_dict[k]
            for k, v in self.transformer.named_parameters()
            if v.requires_grad and k in state_dict
        }

    @abstractmethod
    def predict_velocity(
        self,
        batch: TBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity with the input batch at the given timestep.

        Parameters
        ----------
        batch : dict
            Input batch containing the data.

        timestep : torch.Tensor([B])
            The current timestep. Range is [0, 1], 0 for clean image, 1 for noise.

        Returns
        -------
        torch.Tensor([B, C, H, W])
            The predicted velocity in latent space.
        """
        raise NotImplementedError()

    def _pack_latents(self, latents):
        return rearrange(
            latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def _unpack_latents(self, latents, h, w):
        return rearrange(
            latents,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h // self.patch_size,
            w=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def latent_length_test(self):
        raise NotImplementedError()
