from typing import Any

import torch
import torch.nn as nn
from diffusers import QwenImageTransformer2DModel
from einops import rearrange, repeat

from flow_control.adapters.base import BaseModelAdapter
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class PatchedQwenEmbedRope(nn.Module):
    inv_freq_t: torch.Tensor
    inv_freq_h: torch.Tensor
    inv_freq_w: torch.Tensor

    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        inv_freqs = []
        for dim in axes_dim:
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            inv_freqs.append(inv_freq)
        self.register_buffer("inv_freq_t", inv_freqs[0])
        self.register_buffer("inv_freq_h", inv_freqs[1])
        self.register_buffer("inv_freq_w", inv_freqs[2])

    def _cal_freqs(self, indices: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = torch.outer(indices.to(inv_freq.dtype), inv_freq)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: list[list[tuple[int, int, int]]],
        txt_seq_lens: list[int] | None = None,
        device: torch.device | None = None,
        max_txt_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fhws = video_fhw[0]

        if device is None:
            device = self.inv_freq_t.device

        if txt_seq_lens is None and max_txt_seq_len is None:
            raise ValueError("Either txt_seq_lens or max_txt_seq_len must be provided.")

        vid_freqs_list = []
        max_vid_index = 0

        for idx, (frame, height, width) in enumerate(fhws):
            t_idx = torch.arange(frame, device=device) + idx

            if self.scale_rope:
                h_idx = torch.arange(
                    -(height - height // 2), height // 2, device=device
                )
                w_idx = torch.arange(-(width - width // 2), width // 2, device=device)
                max_vid_index = max(max_vid_index, height // 2, width // 2)
            else:
                h_idx = torch.arange(height, device=device)
                w_idx = torch.arange(width, device=device)
                max_vid_index = max(max_vid_index, height, width)

            freq_t = self._cal_freqs(t_idx, self.inv_freq_t)
            freq_h = self._cal_freqs(h_idx, self.inv_freq_h)
            freq_w = self._cal_freqs(w_idx, self.inv_freq_w)

            video_freq_3d = torch.cat(
                [
                    repeat(freq_t, "f d -> f h w d", h=height, w=width),
                    repeat(freq_h, "h d -> f h w d", f=frame, w=width),
                    repeat(freq_w, "w d -> f h w d", f=frame, h=height),
                ],
                dim=-1,
            )

            vid_freqs_list.append(rearrange(video_freq_3d, "f h w d -> (f h w) d"))

        vid_freqs = torch.cat(vid_freqs_list, dim=0)

        max_len = max_txt_seq_len if max_txt_seq_len is not None else max(txt_seq_lens)  # type: ignore
        txt_idx = torch.arange(max_vid_index, max_vid_index + max_len, device=device)
        txt_freqs = torch.cat(
            [
                self._cal_freqs(txt_idx, self.inv_freq_t),
                self._cal_freqs(txt_idx, self.inv_freq_h),
                self._cal_freqs(txt_idx, self.inv_freq_w),
            ],
            dim=1,
        )

        return vid_freqs, txt_freqs


class BaseQwenImageAdapter(BaseModelAdapter):
    @property
    def transformer(self) -> QwenImageTransformer2DModel:
        return self.hf_model.model  # type: ignore

    @transformer.setter
    def transformer(self, value: Any):
        self.hf_model.model = value

    hf_model: HfModelLoader = HfModelLoader(
        library="diffusers",
        class_name="QwenImageTransformer2DModel",
        pretrained_model_id="Qwen/Qwen-Image",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    def load_transformer(self, use_meta_device=False):
        super().load_transformer(use_meta_device)
        # Replace self.transformer.pos_embed with the above impl
        orig_module = self.transformer.pos_embed
        self.transformer.pos_embed = PatchedQwenEmbedRope(  # type: ignore
            theta=orig_module.theta,
            axes_dim=orig_module.axes_dim,
            scale_rope=orig_module.scale_rope,
        )

    class BatchType(BaseModelAdapter.BatchType):
        prompt_embeds: torch.Tensor
        """`[B, N, D]` Multimodal embeddings from Qwen2.5-VL-7B."""

    def predict_velocity(
        self,
        batch: BatchType,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]

        img_shapes = [[(1, h // 16, w // 16)]] * b

        model_pred = self.transformer(
            hidden_states=batch["noisy_latents"],
            timestep=timestep,
            encoder_hidden_states=batch["prompt_embeds"],
            img_shapes=img_shapes,
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
