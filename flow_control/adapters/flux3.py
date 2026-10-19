"""FLUX 3 Image Klein: per-mode blocks then a joint single stream; Qwen3-VL text, FLUX.2 VAE."""

from typing import Any, Literal, NotRequired

import torch
from diffusers import Flux3Transformer2DModel

from flow_control.adapters.base import BaseModelAdapter, Batch, adapter_registry
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

FLUX3_BASE = "black-forest-labs/flux-3-image-klein-base"
FLUX3_DISTILLED = "black-forest-labs/flux-3-image-klein"
IMAGE_ROPE_SCALE = 0.5
"""The image token grid is RoPE-encoded at half its coordinates (official ``IMAGE_ROPE_SCALE``)."""


class Flux3Batch(Batch):
    prompt_embeds: torch.Tensor
    """`[B, L, 20480]` Eight Qwen3-VL layers concatenated per token, right-padded to a
    multiple of 80. No mask: the DiT attends to the padding tokens as it was trained."""
    reference_latents: NotRequired[list[torch.Tensor]]
    """List of `[B, N_i, 128]` packed, batch-normalized FLUX.2-VAE latents of the references."""
    reference_sizes: NotRequired[list[tuple[int, int]]]
    """List of `(H, W)` pixel sizes of the references."""
    sigma_next: NotRequired[float | torch.Tensor]
    """The enclosing step's target sigma (a float set by ``BaseTransition``; a `[B]`
    tensor once densely collated): the flow map's target timestep. Read by
    ``type="distilled"`` only."""
    _img_ids: NotRequired[torch.Tensor]
    _txt_ids: NotRequired[torch.Tensor]
    _cond_ids: NotRequired[torch.Tensor]


@adapter_registry.register("flux3_base")
@adapter_registry.register("flux3_distilled")
class Flux3Adapter[TBatch: Flux3Batch](
    BaseModelAdapter[Flux3Transformer2DModel, TBatch]
):
    """FLUX 3 Image Klein (5.6B) adapter.

    One class serves both checkpoints: ``type="base"`` is the undistilled model
    (true CFG 3.5, 50 steps) and ``type="distilled"`` the 16-step flow map, which
    needs the timestep each step jumps to (``target_timestep``). The sampler's
    transition puts that on the row as ``sigma_next`` for the duration of the step;
    outside a transition (multistep grids, training points) the distilled forward
    raises.

    Reference images go through the transformer's own reference mode
    (``image_cond_hidden_states``), not the noisy sequence. Position ids are 2-D and
    batch-shared, so a dense group must share ``image_size``, the padded text
    length and the reference layout (the default collator enforces these); rows may
    sit at different steps. ``timestep`` is the sampler's sigma in ``[0, 1]``
    (rescaled by 1000 inside the transformer). ``Flux3PosEmbed`` computes its
    frequencies in ``forward``, so the module is meta-load safe.
    """

    arch: Literal["flux3"] = "flux3"
    type: Literal["base", "distilled"] = "base"

    latent_channels: int = 32
    patch_size: int = 2
    vae_scale_factor: int = 8

    hf_model: HfModelLoader[Flux3Transformer2DModel] = HfModelLoader(
        library="diffusers",
        class_name="Flux3Transformer2DModel",
        pretrained_model_id=FLUX3_BASE,
        subfolder="transformer",
        dtype=torch.bfloat16,
    )

    supports_dense_batching = True
    shared_cache_fields = ("_img_ids", "_txt_ids", "_cond_ids")
    dense_batch_fields = (
        "image_size",
        "noisy_latents",
        "prompt_embeds",
        "reference_latents",
        "reference_sizes",
    )
    # Attention and MLP linears of the 15 per-mode and 28 joint blocks; the
    # embedders, modulation heads and output layer stay frozen.
    lora_target_modules = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.linear_in",
        "ff.linear_out",
    )

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        repo = FLUX3_DISTILLED if self.type == "distilled" else FLUX3_BASE
        official = {FLUX3_BASE, FLUX3_DISTILLED}
        if self.hf_model.pretrained_model_id in official - {repo}:
            # ``type`` picks between the official repos; another id is kept as is.
            self.hf_model = self.hf_model.model_copy(
                update={"pretrained_model_id": repo}
            )

    def _install_modules(self) -> None:
        # The post-load hook: nothing to install, but the checkpoint kind must match.
        flow_map = bool(self.transformer.config["flow_map"])
        if flow_map != (self.type == "distilled"):
            raise ValueError(
                f"flux3 type {self.type!r} expects a "
                f"{'flow-map' if self.type == 'distilled' else 'CFG'} checkpoint, "
                f"but {self.hf_model.pretrained_model_id} has flow_map={flow_map}."
            )

    def _image_ids(self, size: tuple[int, int], l_coord: int = 0) -> torch.Tensor:
        """`(h*w, 4)` ids `(0, row/2, col/2, l)` of one token grid; references use `l = -(1+i)`."""
        stride = self.vae_scale_factor * self.patch_size
        h, w = size[0] // stride, size[1] // stride
        ids = torch.cartesian_prod(
            torch.arange(1), torch.arange(h), torch.arange(w), torch.tensor([l_coord])
        ).float()
        ids[:, 1:3] *= IMAGE_ROPE_SCALE
        return ids.to(self.device)

    def _text_ids(self, length: int) -> torch.Tensor:
        """`(L, 4)` ids `(0, 0, 0, pos)`; the padding tokens count too."""
        ids = torch.zeros(length, 4)
        ids[:, 3] = torch.arange(length)
        return ids.to(self.device)

    def _collate_velocity_inputs(
        self, batches: list[Any], timesteps: list[torch.Tensor]
    ) -> tuple[Any, torch.Tensor] | None:
        """Dense rows may sit at different steps: ``sigma_next`` is collated per sample."""
        collated = super()._collate_velocity_inputs(batches, timesteps)
        if collated is not None and all("sigma_next" in batch for batch in batches):
            collated[0]["sigma_next"] = torch.tensor(
                [batch["sigma_next"] for batch in batches]
            )
        return collated

    def _predict_velocity(self, batch: TBatch, timestep: torch.Tensor) -> torch.Tensor:
        # Singleton ids are cached on the row: the executor may regroup the
        # sample into another physical batch at the next step.
        if "_img_ids" not in batch:
            batch["_img_ids"] = self._image_ids(batch["image_size"])
        if "_txt_ids" not in batch:
            batch["_txt_ids"] = self._text_ids(batch["prompt_embeds"].shape[1])
        kwargs: dict[str, torch.Tensor] = {}
        if batch.get("reference_latents"):
            if "_cond_ids" not in batch:
                batch["_cond_ids"] = torch.cat(
                    [
                        self._image_ids(size, -(1 + index))
                        for index, size in enumerate(batch["reference_sizes"])
                    ]
                )
            kwargs["image_cond_hidden_states"] = torch.cat(
                batch["reference_latents"], dim=1
            )
            kwargs["image_cond_ids"] = batch["_cond_ids"]
        if self.type == "distilled":
            if "sigma_next" not in batch:
                raise ValueError(
                    "flux3 distilled is a flow map and needs the step's target sigma "
                    "(row['sigma_next'], set by a transition node): evaluate it inside "
                    "an euler step, not from a multistep grid or a training point."
                )
            kwargs["target_timestep"] = torch.as_tensor(
                batch["sigma_next"], dtype=timestep.dtype, device=timestep.device
            ).expand(timestep.shape)
        return self.module()(
            hidden_states=batch["noisy_latents"],
            encoder_hidden_states=batch["prompt_embeds"],
            timestep=timestep,
            img_ids=batch["_img_ids"],
            txt_ids=batch["_txt_ids"],
            return_dict=False,
            **kwargs,
        )[0]


if __name__ == "__main__":
    from unittest.mock import patch

    from rich import print

    adapter = Flux3Adapter(type="distilled")
    assert adapter.hf_model.pretrained_model_id == FLUX3_DISTILLED
    with patch.object(Flux3Adapter, "device", torch.device("cpu")):
        ids = adapter._image_ids((64, 48), -2)
        assert ids.shape == (4 * 3, 4)
        # (t, h, w, l): the grid coordinates are halved, references sit at l = -(1+i).
        assert ids[5].tolist() == [0.0, 0.5, 1.0, -2.0]
        text = adapter._text_ids(80)
        assert text[79].tolist() == [0.0, 0.0, 0.0, 79.0]
    print("[green]flux3 ids ok[/green]")
