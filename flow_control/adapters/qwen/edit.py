import torch

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

from .base import BaseQwenImageAdapter

logger = get_logger(__name__)


class QwenImageEditAdapter(BaseQwenImageAdapter):
    hf_model: HfModelLoader = HfModelLoader(
        type="diffusers",
        class_name="QwenImageTransformer2DModel",
        pretrained_model_id="Qwen/Qwen-Image-Edit-2509",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    class BatchType(BaseQwenImageAdapter.BatchType):
        reference_latents: list[torch.Tensor]
        """List of `[B, N, D]` Tensors representing VAE encoded reference images."""
        reference_sizes: list[tuple[int, int]]
        """List of `(H, W)` tuples representing the sizes of the reference images."""

    def predict_velocity(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]

        input_latents = torch.cat(
            [batch["noisy_latents"]] + batch["reference_latents"], dim=1
        )

        img_shapes = [
            [
                (1, h // 16, w // 16),
                *[
                    (1, size[0] // 16, size[1] // 16)
                    for size in batch["reference_sizes"]
                ],
            ]
        ] * b
        txt_seq_lens = [batch["prompt_embeds"].shape[1]] * b

        model_pred = self.transformer(
            hidden_states=input_latents,
            timestep=timestep / 1000,
            encoder_hidden_states=batch["prompt_embeds"],
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        return model_pred[:, :n, :]
