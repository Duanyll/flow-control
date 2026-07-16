from typing import Any, Literal, NotRequired, cast

import torch
from peft import LoraConfig

from flow_control.adapters.base import BaseModelAdapter, Batch, adapter_registry
from flow_control.third_party.hidream_o1 import get_rope_index_fix_point
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import deep_cast_float_dtype, deep_move_to_device

from .modeling import HiDreamO1Transformer

logger = get_logger(__name__)

HIDREAM_O1_FULL = "HiDream-ai/HiDream-O1-Image"
HIDREAM_O1_DEV = "HiDream-ai/HiDream-O1-Image-Dev"


class HiDreamO1Batch(Batch):
    input_ids: torch.Tensor
    """`[1, L]` Prompt token ids: chat template (+ expanded VLM image placeholders
    for editing) + `<|boi_token|>` + `<|tms_token|>`. The trailing target/reference
    vision tokens are resolution-dependent and appended by the adapter."""
    pixel_values: NotRequired[torch.Tensor]
    """VLM thumbnail patches for the SigLIP condition path (editing only)."""
    image_grid_thw: NotRequired[torch.Tensor]
    """`[K, 3]` VLM thumbnail grids matching ``pixel_values`` (editing only)."""
    reference_latents: NotRequired[list[torch.Tensor]]
    """Per-reference packed pixel latents `[1, N_r, 3072]` (editing only)."""
    reference_sizes: NotRequired[list[tuple[int, int]]]
    """Per-reference `(H, W)` pixel sizes (editing only)."""


@adapter_registry.register("hidream_full")
@adapter_registry.register("hidream_dev")
class HiDreamO1Adapter[TBatch: HiDreamO1Batch](
    BaseModelAdapter[HiDreamO1Transformer, TBatch]
):
    """HiDream-O1 pixel-space unified-transformer adapter.

    One class serves both checkpoints: ``type="full"`` loads the undistilled
    model (true CFG 5.0, 50-step UniPC) and ``type="dev"`` the 28-step distilled
    one (guidance-free, flash solver) — sampler concerns, so the forward body is
    identical. The checkpoint is selected in :meth:`model_post_init`.

    Geometry: pixels are the latents (``vae_scale_factor=1``) patchified 32x32,
    so the framework's packed ``[B, HW/1024, 3072]`` layout *is* the model's
    ``vinputs`` layout. Latents are scaled pixels (``IdentityVAE`` with
    ``scaling=1/8`` absorbs the model's noise-scale-8 forward process into a
    unit-noise flow); the adapter multiplies by ``1/latent_scaling`` before the
    model and by ``latent_scaling`` after.

    Conventions translated in :meth:`_predict_velocity`:

    - model time runs 0(noise) -> 1(clean), i.e. ``1 - sigma`` (embedder
      multiplies by 1000 internally);
    - the model predicts x0 (``x_pred``); framework velocity is
      ``(z - x0) / clamp(sigma, 1e-3)`` (official ``T_EPS`` clamp);
    - the timestep stays float32 end-to-end (``TimestepEmbedder`` self-casts;
      bf16 would collapse ``1 - sigma`` near the noise end).
    """

    arch: Literal["hidream"] = "hidream"
    type: Literal["full", "dev"] = "full"

    latent_channels: int = 3
    patch_size: int = 32
    vae_scale_factor: int = 1

    latent_scaling: float = 0.125
    """Pixel -> latent scale; must match the preset's ``IdentityVAE.scaling``."""
    sigma_eps: float = 1e-3
    """Sigma clamp for the x0 -> velocity conversion (official ``T_EPS``)."""
    num_timestep_tokens: int = 1
    """Trailing `<|tms_token|>` count in ``input_ids`` (official uses 1)."""
    use_flash_attn: bool = False
    """Use the flash-attention two-pass hybrid-attention path (requires
    flash-attn installed). ``False`` uses the mathematically equivalent dense
    4D-mask SDPA path."""

    hf_model: HfModelLoader[HiDreamO1Transformer] = HfModelLoader(
        library="custom",
        class_name="flow_control.adapters.hidream.modeling.HiDreamO1Transformer",
        pretrained_model_id=HIDREAM_O1_FULL,
        dtype=torch.bfloat16,
    )
    peft_lora_config: LoraConfig = LoraConfig(
        # Language-model attention projections. The vision tower uses different
        # module names (qkv/proj), so these do not touch the SigLIP path.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        repo = HIDREAM_O1_DEV if self.type == "dev" else HIDREAM_O1_FULL
        if self.hf_model.pretrained_model_id != repo:
            # Fresh copy so we never mutate a shared field default.
            self.hf_model = self.hf_model.model_copy(
                update={"pretrained_model_id": repo}
            )

    def predict_velocity(self, batch: TBatch, timestep: torch.Tensor) -> torch.Tensor:
        # Same as the base implementation except the timestep is kept float32:
        # near sigma=1 the model time 1-sigma underflows bf16 resolution.
        batch = deep_cast_float_dtype(batch, self.dtype)
        batch = deep_move_to_device(batch, self.device)
        timestep = timestep.to(device=self.device, dtype=torch.float32)
        return self._predict_velocity(batch, timestep).float()

    def _build_sequence(
        self, batch: TBatch, txt_len: int, grid: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append target/reference vision tokens and derive the mRoPE index.

        Mirrors the official ``build_t2i_text_sample`` / editing branch: each
        vision segment is ``[vision_start, image_token * (n-1)]``; the rope index
        gets VLM thumbnail grids (divided by ``spatial_merge_size``) first, then
        the target grid, then reference grids; ``token_types`` marks the timestep
        token(s) and every vision token as full-attention generation tokens.

        Returns ``(input_ids_pad, position_ids, token_types)``.
        """
        # The vendored modeling module is excluded from type checking, so the
        # nested Qwen3VLConfig attributes are opaque to pyright.
        config: Any = self.transformer.config
        device = self.device
        input_ids = batch["input_ids"]

        ref_sizes = batch.get("reference_sizes") or []
        ref_grids = [
            (
                rh // (self.vae_scale_factor * self.patch_size),
                rw // (self.vae_scale_factor * self.patch_size),
            )
            for rh, rw in ref_sizes
        ]

        segments = [grid, *ref_grids]
        vision_tokens = []
        for seg_h, seg_w in segments:
            seg = torch.full(
                (1, seg_h * seg_w),
                config.image_token_id,
                dtype=input_ids.dtype,
                device=device,
            )
            seg[0, 0] = config.vision_start_token_id
            vision_tokens.append(seg)
        input_ids_pad = torch.cat([input_ids, *vision_tokens], dim=-1)

        grid_rows = torch.tensor(
            [[1, seg_h, seg_w] for seg_h, seg_w in segments],
            dtype=torch.int64,
            device=device,
        )
        if "image_grid_thw" in batch:
            # VLM thumbnail grids are in vision-tower patch units; the rope index
            # wants them in merged-token units.
            merge = config.vision_config.spatial_merge_size
            thumb_grids = batch["image_grid_thw"].clone().to(device)
            thumb_grids[:, 1] //= merge
            thumb_grids[:, 2] //= merge
            grid_thw = torch.cat([thumb_grids, grid_rows], dim=0)
            skip_vision_start = [0] * thumb_grids.shape[0] + [1] * len(segments)
        else:
            grid_thw = grid_rows
            skip_vision_start = [1] * len(segments)

        position_ids, _ = get_rope_index_fix_point(
            1,  # grids above are already in final token units
            config.image_token_id,
            config.video_token_id,
            config.vision_start_token_id,
            input_ids=cast(torch.LongTensor, input_ids_pad),
            image_grid_thw=cast(torch.LongTensor, grid_thw),
            video_grid_thw=None,
            attention_mask=None,
            skip_vision_start_token=skip_vision_start,
        )

        token_types = torch.zeros_like(input_ids_pad)
        token_types[0, txt_len - self.num_timestep_tokens :] = 1

        return input_ids_pad, position_ids, token_types

    def _predict_velocity(
        self,
        batch: TBatch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        noisy_latents = batch["noisy_latents"]
        if noisy_latents.shape[0] != 1:
            raise ValueError("HiDreamO1Adapter only supports batch size 1.")

        h, w = batch["image_size"]
        scale = self.vae_scale_factor * self.patch_size
        grid = (h // scale, w // scale)
        tgt_len = grid[0] * grid[1]
        txt_len = batch["input_ids"].shape[1]

        _, position_ids, token_types = self._build_sequence(batch, txt_len, grid)

        vinputs = noisy_latents / self.latent_scaling
        reference_latents = batch.get("reference_latents") or []
        if reference_latents:
            vinputs = torch.cat(
                [vinputs, *(ref / self.latent_scaling for ref in reference_latents)],
                dim=1,
            )

        sigma = timestep.float().reshape(-1)
        outputs = self.transformer(
            input_ids=batch["input_ids"],
            position_ids=position_ids,
            vinputs=vinputs.to(self.dtype),
            timestep=1.0 - sigma,
            token_types=token_types,
            use_flash_attn=self.use_flash_attn,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
        )
        # Sequence layout is [text | target | references]; the target rows are
        # contiguous right after the text.
        x_pred = outputs.x_pred[:, txt_len : txt_len + tgt_len]

        x0_latents = x_pred.float() * self.latent_scaling
        sigma = sigma.clamp_min(self.sigma_eps).view(-1, 1, 1)
        return (noisy_latents.float() - x0_latents) / sigma
