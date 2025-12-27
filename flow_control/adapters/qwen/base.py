from typing import Any, NotRequired

import torch
from diffusers import QwenImageTransformer2DModel

from flow_control.adapters.base import BaseModelAdapter
from flow_control.utils.loaders import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


_qwen_embed_rope_patched = False


def patch_qwen_embed_rope():
    global _qwen_embed_rope_patched
    if _qwen_embed_rope_patched:
        return
    from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

    original_forward = QwenEmbedRope.forward

    def patched_forward(self, video_fhw, txt_seq_lens, device):
        if self.pos_freqs.device.type == "meta":
            pos_index = torch.arange(4096)
            neg_index = torch.arange(4096).flip(0) * -1 - 1
            self.pos_freqs = torch.cat(
                [
                    self.rope_params(pos_index, self.axes_dim[0], self.theta),
                    self.rope_params(pos_index, self.axes_dim[1], self.theta),
                    self.rope_params(pos_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            )
            self.neg_freqs = torch.cat(
                [
                    self.rope_params(neg_index, self.axes_dim[0], self.theta),
                    self.rope_params(neg_index, self.axes_dim[1], self.theta),
                    self.rope_params(neg_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            )
        return original_forward(self, video_fhw, txt_seq_lens, device)

    QwenEmbedRope.forward = patched_forward
    logger.info("Patched QwenEmbedRope.forward to handle meta device.")
    _qwen_embed_rope_patched = True


class BaseQwenImageAdapter(BaseModelAdapter):
    @property
    def transformer(self) -> QwenImageTransformer2DModel:
        return self._transformer  # type: ignore

    @transformer.setter
    def transformer(self, value: Any):
        self._transformer = value

    hf_model: HfModelLoader = HfModelLoader(
        type="diffusers",
        class_name="QwenImageTransformer2DModel",
        pretrained_model_id="Qwen/Qwen-Image",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    def load_transformer(self, use_meta_device=False):
        patch_qwen_embed_rope()
        return super().load_transformer(use_meta_device)

    class BatchType(BaseModelAdapter.BatchType):
        prompt_embeds: torch.Tensor
        """`[B, N, D]` Multimodal embeddings from Qwen2.5-VL-7B."""
        prompt_embeds_mask: NotRequired[torch.Tensor]
        # TODO: What shape is this?

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

        img_shapes = [[(1, h // 16, w // 16)]] * b
        txt_seq_lens = batch["prompt_embeds_mask"].sum(dim=1).tolist()

        model_pred = self.transformer(
            hidden_states=batch["noisy_latents"],
            timestep=timestep / 1000,
            encoder_hidden_states_mask=batch["prompt_embeds_mask"],
            encoder_hidden_states=batch["prompt_embeds"],
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        return model_pred

    def _make_attention_mask(self, prompt_embeds):
        b, n, d = prompt_embeds.shape
        return torch.ones(
            (b, n),
            dtype=torch.long,
            device=prompt_embeds.device,
        )
