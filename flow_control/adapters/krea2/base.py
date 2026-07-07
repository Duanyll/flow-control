from typing import Literal

import torch
from diffusers import Krea2Transformer2DModel
from einops import rearrange
from peft import LoraConfig

from flow_control.adapters.base import BaseModelAdapter, Batch, adapter_registry
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

KREA2_RAW = "krea/Krea-2-Raw"
KREA2_TURBO = "krea/Krea-2-Turbo"


class Krea2Batch(Batch):
    prompt_embeds: torch.Tensor
    """`[B, N, num_text_layers, D]` Stacked Qwen3-VL hidden states (12 tapped layers)."""
    prompt_embeds_mask: torch.Tensor
    """`[B, N]` Boolean text attention mask (padding kept, marked ``False``)."""


@adapter_registry.register("krea2_base")
@adapter_registry.register("krea2_turbo")
class Krea2Adapter[TBatch: Krea2Batch](
    BaseModelAdapter[Krea2Transformer2DModel, TBatch]
):
    """Krea 2 single-stream DiT adapter.

    One class serves both checkpoints: ``type="base"`` loads Krea-2-Raw (undistilled,
    true CFG) and ``type="turbo"`` loads Krea-2-Turbo (8-step distilled, guidance-free —
    a sampler concern, so the forward body is identical). The checkpoint is selected in
    :meth:`model_post_init`.

    Unlike the Qwen adapter, Krea's ``Krea2RotaryPosEmbed`` computes its frequencies
    lazily in ``forward`` (no ``__init__`` tensors), so it is meta-load safe and needs no
    rope monkey-patch. The transformer consumes the packed latents directly plus a
    combined ``position_ids`` (text rows zero, image rows carry the latent-grid ``(0,h,w)``)
    and the text ``encoder_attention_mask``. ``timestep`` is the sampler's sigma in
    ``[0, 1]`` (the transformer rescales by 1000 internally).
    """

    arch: Literal["krea2"] = "krea2"
    type: Literal["base", "turbo"] = "base"

    latent_channels: int = 16
    patch_size: int = 2
    vae_scale_factor: int = 8

    hf_model: HfModelLoader[Krea2Transformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="Krea2Transformer2DModel",
        pretrained_model_id=KREA2_RAW,
        subfolder="transformer",
        dtype=torch.bfloat16,
    )
    peft_lora_config: LoraConfig = LoraConfig(
        # Attention projections of the main DiT blocks (Krea2Attention). NOTE: these
        # suffixes also match the Krea2TextFusion attention; refine when we build the RL
        # training config (this pass is inference-only, so LoRA is not exercised yet).
        target_modules=[
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_gate",
            "attn.to_out.0",
        ],
    )

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        repo = KREA2_TURBO if self.type == "turbo" else KREA2_RAW
        if self.hf_model.pretrained_model_id != repo:
            # Fresh copy so we never mutate a shared field default.
            self.hf_model = self.hf_model.model_copy(
                update={"pretrained_model_id": repo}
            )

    def _prepare_position_ids(
        self, text_seq_len: int, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Combined `(text_seq_len + grid_h * grid_w, 3)` mRoPE coordinates: text rows at
        the origin, image rows carrying their `(0, h, w)` latent-grid position."""
        device = self.device
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = torch.zeros(grid_h, grid_w, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        image_ids = rearrange(image_ids, "h w c -> (h w) c")
        return torch.cat([text_ids, image_ids], dim=0)

    def _predict_velocity(
        self,
        batch: TBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        h, w = batch["image_size"]
        grid_h = h // (self.vae_scale_factor * self.patch_size)
        grid_w = w // (self.vae_scale_factor * self.patch_size)
        text_seq_len = batch["prompt_embeds"].shape[1]
        position_ids = self._prepare_position_ids(text_seq_len, grid_h, grid_w)

        model_pred = self.transformer(
            hidden_states=batch["noisy_latents"],
            encoder_hidden_states=batch["prompt_embeds"],
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=batch["prompt_embeds_mask"],
            return_dict=False,
        )[0]

        return model_pred
