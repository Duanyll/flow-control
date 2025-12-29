import torch

from flow_control.utils.loaders import HfModelLoader
from flow_control.utils.logging import get_logger

from .base import BaseQwenImageAdapter

logger = get_logger(__name__)


class QwenImageLayeredAdapter(BaseQwenImageAdapter):
    hf_model: HfModelLoader = HfModelLoader(
        type="diffusers",
        class_name="QwenImageTransformer2DModel",
        pretrained_model_id="Qwen/Qwen-Image-Layered",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    class BatchType(BaseQwenImageAdapter.BatchType):
        num_layers: int
        """Number of layers in the layered image generation."""
        image_latents: torch.Tensor

    def predict_velocity(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]

        if "prompt_embeds_mask" not in batch:
            batch["prompt_embeds_mask"] = self._make_attention_mask(
                batch["prompt_embeds"]
            )

        input_latents = torch.cat(
            [batch["noisy_latents"], batch["image_latents"]], dim=1
        )

        img_shapes = [[(1, h // 16, w // 16)] * (batch["num_layers"] + 2)] * b
        txt_seq_lens = batch["prompt_embeds_mask"].sum(dim=1).tolist()

        model_pred = self.transformer(
            hidden_states=input_latents,
            timestep=timestep / 1000,
            encoder_hidden_states_mask=batch["prompt_embeds_mask"],
            encoder_hidden_states=batch["prompt_embeds"],
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        return model_pred
