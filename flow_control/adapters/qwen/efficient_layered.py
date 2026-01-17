from typing import Any, Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from flow_control.utils.logging import get_logger

from .base import BaseQwenImageAdapter

logger = get_logger(__name__)


class EfficientLayeredQwenEmbedRope(nn.Module):
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
        video_fhw: Any,
        txt_seq_lens: list[int] | None = None,
        device: torch.device | None = None,
        max_txt_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(video_fhw, tuple):
            video_fhw, txt_seq_lens = video_fhw
        if txt_seq_lens is None:
            txt_seq_lens = [max_txt_seq_len] # type: ignore

        fhws = video_fhw[0]

        if device is None:
            device = self.inv_freq_t.device

        vid_freqs_list = []
        max_vid_index = 0

        _, all_height, all_width = fhws[0]

        for idx, spec in enumerate(fhws):
            if len(spec) == 3:
                frame, height, width = spec
                t_idx = torch.arange(frame, device=device) + idx

                if self.scale_rope:
                    h_idx = torch.arange(
                        -(height - height // 2), height // 2, device=device
                    )
                    w_idx = torch.arange(
                        -(width - width // 2), width // 2, device=device
                    )
                    max_vid_index = max(max_vid_index, height // 2, width // 2)
                else:
                    h_idx = torch.arange(height, device=device)
                    w_idx = torch.arange(width, device=device)
                    max_vid_index = max(max_vid_index, height, width)
            else:
                frame, top, bottom, left, right = spec
                if self.scale_rope:
                    top -= all_height // 2
                    bottom -= all_height // 2
                    left -= all_width // 2
                    right -= all_width // 2
                t_idx = torch.arange(frame, device=device) + idx
                h_idx = torch.arange(top, bottom, device=device)
                w_idx = torch.arange(left, right, device=device)
                max_vid_index = max(
                    max_vid_index, abs(top), abs(bottom), abs(left), abs(right)
                )
                height = bottom - top
                width = right - left

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

        txt_idx = torch.cat(
            [
                torch.arange(max_vid_index, max_vid_index + length, device=device)
                for length in txt_seq_lens # type: ignore
            ]
        )
        txt_freqs = torch.cat(
            [
                self._cal_freqs(txt_idx, self.inv_freq_t),
                self._cal_freqs(txt_idx, self.inv_freq_h),
                self._cal_freqs(txt_idx, self.inv_freq_w),
            ],
            dim=1,
        )

        return vid_freqs, txt_freqs


class EfficientLayeredQwenImageAdapter(BaseQwenImageAdapter):
    attn_mask_mode: Literal["full", "per-layer", "per-layer-strict"] = "per-layer"
    """
    Attention mask mode for the adapter. Options are:
    - "full": No attention masking, full attention.
    - "per-layer": Each layer attends only to themselves, their corresponding text prompt,
      and the base image. They do not attend to other layers.
    - "per-layer-strict": Similar to "per-layer", but the base image does not attend to
      any layers, ensuring the decoupling of layers from the base image.
    """
    attn_block_size: int = 128

    class BatchType(BaseQwenImageAdapter.BatchType):
        image_latents: torch.Tensor
        layer_boxes: list[tuple[int, int, int, int]]
        """
        `(top, bottom, left, right)` in pixels for each layer in the image. Should be aligned with
        multiples of 16. `top` and `left` are inclusive, `bottom` and `right` are exclusive.
        """
        text_lengths: list[int]
        """
        Lengths of prompts corresponding to each layer in the image.
        """

    def _get_compiled_create_block_mask(self):
        if hasattr(self, "_compiled_create_block_mask"):
            return self._compiled_create_block_mask
        else:
            logger.info("Compiling create_block_mask function for the first time...")
            compiled_fn = torch.compile(create_block_mask)
            self._compiled_create_block_mask = compiled_fn
            return compiled_fn

    def load_transformer(self, use_meta_device=False):
        super().load_transformer(use_meta_device)
        self.transformer.set_attention_backend("flex")
        orig_module = self.transformer.pos_embed
        self.transformer.pos_embed = EfficientLayeredQwenEmbedRope(  # type: ignore
            theta=orig_module.theta,
            axes_dim=orig_module.axes_dim,
            scale_rope=orig_module.scale_rope,
        )

    def make_block_mask(
        self, base_len: int, layer_lens: list[int], txt_lens: list[int]
    ) -> BlockMask | None:
        if self.attn_mask_mode == "full":
            return None
        elif self.attn_mask_mode in ["per-layer", "per-layer-strict"]:
            total_len = base_len + sum(layer_lens) + sum(txt_lens)
            layer_ids = torch.zeros(total_len, dtype=torch.long, device=self.device)
            current_loc_layer = base_len
            current_loc_txt = base_len + sum(layer_lens)
            for i, (layer_size, txt_size) in enumerate(
                zip(layer_lens, txt_lens, strict=True)
            ):
                layer_ids[current_loc_layer : current_loc_layer + layer_size] = i + 1
                current_loc_layer += layer_size
                layer_ids[current_loc_txt : current_loc_txt + txt_size] = i + 1
                current_loc_txt += txt_size

            def per_layer_mask_fn(b, h, q_idx, kv_idx):
                return (
                    (layer_ids[q_idx] == layer_ids[kv_idx])
                    | (layer_ids[q_idx] == 0)
                    | (layer_ids[kv_idx] == 0)
                )

            def per_layer_strict_mask_fn(b, h, q_idx, kv_idx):
                return (layer_ids[q_idx] == layer_ids[kv_idx]) | (
                    layer_ids[kv_idx] == 0
                )

            mask_fn = (
                per_layer_mask_fn
                if self.attn_mask_mode == "per-layer"
                else per_layer_strict_mask_fn
            )

            block_mask = self._get_compiled_create_block_mask()(
                mask_fn,
                1,
                1,
                total_len,
                total_len,
                BLOCK_SIZE=self.attn_block_size,
                device=self.device,
            )
            return block_mask
        else:
            raise ValueError(f"Unknown attn_mask_mode: {self.attn_mask_mode}")

    def predict_velocity(self, batch: BatchType, timestep):
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]

        input_latents = torch.cat(
            [batch["image_latents"], batch["noisy_latents"]], dim=1
        )

        img_shapes = [
            [1, h // 16, w // 16]
            + [
                (1, top // 16, bottom // 16, left // 16, right // 16)
                for (top, bottom, left, right) in batch["layer_boxes"]
            ]
        ] * b
        txt_seq_lens = batch["text_lengths"]

        block_mask = self.make_block_mask(
            base_len=batch["image_latents"].shape[1],
            layer_lens=[
                (bottom - top) * (right - left) // 256
                for (top, bottom, left, right) in batch["layer_boxes"]
            ],
            txt_lens=batch["text_lengths"],
        )

        model_pred = self.transformer(
            hidden_states=input_latents,
            timestep=timestep,
            encoder_hidden_states=batch["prompt_embeds"],
            img_shapes=(img_shapes, txt_seq_lens),
            attention_kwargs={
                "attention_mask": block_mask,
            },
            return_dict=False,
        )[0]

        return model_pred[:, batch["image_latents"].shape[1] :, :]
