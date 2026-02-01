from abc import ABC, abstractmethod
from typing import Any, TypedDict

import torch
from einops import rearrange
from pydantic import BaseModel, ConfigDict

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.types import TorchDType
from flow_control.utils.upcasting import (
    apply_layerwise_upcasting,
    cast_trainable_parameters,
)

logger = get_logger(__name__)


class BaseModelAdapter(BaseModel, ABC):
    """
    Base class for all control adapters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def transformer(self) -> Any:
        return self.hf_model.model

    @transformer.setter
    def transformer(self, value: Any):
        self.hf_model.model = value

    @property
    def device(self) -> torch.device:
        return self.transformer.device  # type: ignore

    hf_model: HfModelLoader
    storage_dtype: TorchDType | None = None
    trainable_dtype: TorchDType = torch.bfloat16
    all_trainable: bool = False

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

    class BatchType(TypedDict):
        image_size: tuple[int, int]
        """`(H, W)` The size of the image to generate."""
        clean_latents: torch.Tensor
        """`[B, N, D]` The clean latents. Only available during training."""
        noisy_latents: torch.Tensor
        """`[B, N, D]` The noisy latents to denoise."""

    def load_transformer(self, device: torch.device) -> None:
        self.transformer = self.hf_model.load_model(device=device)  # type: ignore
        self.transformer.requires_grad_(self.all_trainable)
        self._install_modules()
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
        created.
        """
        pass

    def accelerate_save_model(self) -> dict:
        """
        Decide which layers to save in the checkpoint. Will be wrapped and registered by
        `accelerator.register_save_state_pre_hook`.

        :param transformer: The adapted transformer model.
        :return: A state_dict containing the layers to save.
        """
        if self.all_trainable:
            return self.transformer.state_dict()
        else:
            return {}

    def accelerate_load_model(self, state_dict: dict):
        """
        Load the state_dict. Will be wrapped and registered by `accelerator.register_load_state_pre_hook`.

        :param transformer: The adapted transformer model.
        :param state_dict: The state_dict containing the layers to load.
        """
        self.transformer.load_state_dict(state_dict, strict=False)

    def filter_state_dict(self, state_dict: dict) -> dict:
        """
        Filter the state_dict before saving. May be called with FSDP mangled keys, so
        avoid precise key matching but look for substrings. Do not modify the original
        state_dict in-place so subclasses can choose to call super().

        :param state_dict: The original state_dict.
        :return: The filtered state_dict.
        """
        if self.all_trainable:
            return state_dict
        else:
            return {}

    @abstractmethod
    def predict_velocity(
        self,
        batch: BatchType,
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
