from typing import TypedDict, Any

import torch
from pydantic import BaseModel, ConfigDict


class BaseModelAdapter(BaseModel):
    """
    Base class for all control adapters.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _transformer: Any

    @property
    def transformer(self) -> Any:
        return self._transformer
    
    @transformer.setter
    def transformer(self, value: Any):
        self._transformer = value

    @property
    def device(self) -> torch.device:
        return self.transformer.device # type: ignore
    
    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.dtype # type: ignore

    class BatchType(TypedDict):
        clean_latents: torch.Tensor
        """`[B, C, H, W]` The clean latents. Only available during training."""
        noisy_latents: torch.Tensor
        """`[B, C, H, W]` The noisy latents to denoise."""

    def load_transformer(self):
        pass

    def _install_modules(self):
        """
        Create and initialize additional modules on the base model. Called after base model is
        created.
        """
        pass

    def save_model(self) -> dict:
        """
        Decide which layers to save in the checkpoint. Will be wrapped and registered by
        `accelerator.register_save_state_pre_hook`.

        :param transformer: The adapted transformer model.
        :return: A state_dict containing the layers to save.
        """
        return {}

    def load_model(self, state_dict: dict):
        """
        Load the state_dict. Will be wrapped and registered by `accelerator.register_load_state_pre_hook`.

        :param transformer: The adapted transformer model.
        :param state_dict: The state_dict containing the layers to load.
        """
        pass

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
    
    def generate_noise(
        self,
        batch: BatchType,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError()
    
    def get_latent_length(self, batch: BatchType) -> int:
        raise NotImplementedError()

    def train_step(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run forward pass and compute loss for the given batch. May call `predict_velocity`
        internally.

        Parameters
        ----------
        batch : dict
            Input batch containing the data.
        timestep : torch.Tensor([B])
            The current timestep. Range is [0, 1], 0 for clean image, 1 for noise.

        Returns
        -------
        torch.Tensor([B])
            Unweighted loss for each sample in the batch.
        """
        raise NotImplementedError()
    