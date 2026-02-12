import torch
import torch.nn as nn
from diffusers import QwenImageTransformer2DModel
from einops import rearrange, repeat

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

from ..base import BaseModelAdapter
from .base import QwenImageAdapter, QwenImageBatch

logger = get_logger(__name__)


class QwenImageLayeredBatch(QwenImageBatch):
    num_layers: int
    """Number of layers in the layered image generation."""
    image_latents: torch.Tensor
    """`[B, N, D]` Tensor representing input image latents."""


class PatchedQwenEmbedLayer3DRope(nn.Module):
    """
    Reimplementation of QwenEmbedLayer3DRope from diffusers to correctly utilize nn.Module' buffer system.
    Supports meta device loading and efficient caching.
    """

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
            # Precompute the inverse frequencies
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
        max_txt_seq_len: int | torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # video_fhw structure: [Batch_List[Layer_Tuple(f, h, w), ...]]
        # We process the first item in the batch (assuming batch consistency or BS=1)
        fhws = video_fhw[0]

        if device is None:
            device = self.inv_freq_t.device

        vid_freqs_list = []
        max_vid_index = 0
        layer_num = len(fhws) - 1

        for idx, (frame, height, width) in enumerate(fhws):
            # 1. Calculate Temporal Indices
            if idx != layer_num:
                # Standard layer: time grows from idx
                t_idx = torch.arange(frame, device=device) + idx
            else:
                # Condition layer (Last item):
                # Original logic uses neg_freqs[-1], which corresponds to index -1.
                # Even if frame > 1, the condition image shares the same time index -1 across all its frames.
                t_idx = torch.full((frame,), -1.0, device=device)

            # 2. Calculate Spatial Indices
            if self.scale_rope:
                h_idx = torch.arange(
                    -(height - height // 2), height // 2, device=device
                )
                w_idx = torch.arange(-(width - width // 2), width // 2, device=device)
                current_max_dim = max(height // 2, width // 2)
            else:
                h_idx = torch.arange(height, device=device)
                w_idx = torch.arange(width, device=device)
                current_max_dim = max(height, width)

            max_vid_index = max(max_vid_index, current_max_dim)

            # 3. Compute Frequencies
            freq_t = self._cal_freqs(t_idx, self.inv_freq_t)
            freq_h = self._cal_freqs(h_idx, self.inv_freq_h)
            freq_w = self._cal_freqs(w_idx, self.inv_freq_w)

            # 4. Broadcast and Combine (Time, Height, Width)
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

        # 5. Compute Text Frequencies
        # Ensure max_vid_index is at least layer_num (from original logic)
        max_vid_index = max(max_vid_index, layer_num)

        max_len = int(max_txt_seq_len)
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


class QwenImageLayeredAdapter(QwenImageAdapter[QwenImageLayeredBatch]):
    hf_model: HfModelLoader[QwenImageTransformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="QwenImageTransformer2DModel",
        pretrained_model_id="Qwen/Qwen-Image-Layered",
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    def load_transformer(self, device: torch.device) -> None:
        BaseModelAdapter.load_transformer(self, device=device)
        # Replace self.transformer.pos_embed with the above impl
        orig_module = self.transformer.pos_embed
        self.transformer.pos_embed = PatchedQwenEmbedLayer3DRope(  # type: ignore
            theta=orig_module.theta,
            axes_dim=orig_module.axes_dim,
            scale_rope=orig_module.scale_rope,
        )

    def predict_velocity(
        self,
        batch: QwenImageLayeredBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]

        input_latents = torch.cat(
            [batch["noisy_latents"], batch["image_latents"]], dim=1
        )

        img_shapes = [[(1, h // 16, w // 16)] * (batch["num_layers"] + 2)] * b

        is_rgb = torch.tensor([0] * b).to(device=self.device, dtype=torch.long)

        model_pred = self.transformer(
            hidden_states=input_latents,
            timestep=timestep,
            encoder_hidden_states=batch["prompt_embeds"],
            img_shapes=img_shapes,
            return_dict=False,
            additional_t_cond=is_rgb,
        )[0]

        return model_pred[:, :n, :]
