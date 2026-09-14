"""Whole-image constraints before a transition and after guidance combine."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

import torch
from einops import repeat
from pydantic import BaseModel, ConfigDict

from flow_control.adapters.base import Batch
from flow_control.utils.condition_image import ConditionImage, ConditionImageSpec
from flow_control.utils.registry import Registry, RegistryUnion

from .plan import EvalRequest, StepContext, Transition


class BaseProjector(BaseModel):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    def pre_transition(
        self,
        latents: torch.Tensor,
        row: Batch,
        ctx: StepContext,
        transition: Transition,
    ) -> torch.Tensor:
        return latents

    def post_combine(
        self,
        velocity: torch.Tensor,
        request: EvalRequest,
        row: Batch,
        ctx: StepContext,
    ) -> torch.Tensor:
        """Project x0 = request.latents - request.sigma * velocity.

        Implementations return the corrected velocity. The identity keeps the
        original tensor, avoiding a lossy x0 round-trip for pre-only projectors.
        """
        return velocity


projector_registry: Registry[BaseProjector] = Registry("projector", base=BaseProjector)
Projector = Annotated[BaseProjector, RegistryUnion(projector_registry, "type")]


def apply_pre_transition(
    projectors: Sequence[BaseProjector],
    row: Batch,
    ctx: StepContext,
    transition: Transition,
) -> torch.Tensor:
    latents = ctx.latents
    for projector in projectors:
        latents = projector.pre_transition(latents, row, ctx, transition)
    return latents


def _row_tensor(row: Batch, field: str) -> torch.Tensor:
    value = cast(dict[str, Any], row).get(field)
    if not isinstance(value, torch.Tensor):
        raise KeyError(
            f"Differential diffusion requires tensor field {field!r}; available fields: {sorted(row)}."
        )
    return value


@projector_registry.register("differential")
class DifferentialDiffusion(BaseProjector):
    """Release pixels according to an edit-strength map over executed steps.

    White pixels are editable immediately, black pixels follow the noised
    reference, and intermediate strengths release pixels later in the run.
    Masks and reference latents describe the whole image, including with tiles.
    """

    type: Literal["differential"] = "differential"
    source: ConditionImageSpec = "inpaint"
    """Condition image whose latents the mask keeps, e.g. ``"inpaint"`` or
    ``"reference[0]"``; it must match the generated latent geometry."""

    @staticmethod
    def _edit_strength(row: Batch, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError(
                f"Differential diffusion requires packed BND latents, got {tuple(latents.shape)}."
            )
        mask = _row_tensor(row, "inpaint_mask_latents")
        if mask.ndim != 3:
            raise ValueError(
                f"Differential diffusion expects inpaint_mask_latents in BND format, got {tuple(mask.shape)}."
            )
        if mask.shape[:2] != latents.shape[:2]:
            raise ValueError(
                "Differential diffusion requires the whole-image mask to match latent batch and token dimensions; "
                f"got {tuple(mask.shape)} and {tuple(latents.shape)}."
            )
        patch_features = mask.shape[2]
        if patch_features == 0 or latents.shape[2] % patch_features:
            raise ValueError(
                f"Differential diffusion cannot expand packed mask features {patch_features} over latent features {latents.shape[2]}."
            )
        return repeat(
            mask, "b n p -> b n (c p)", c=latents.shape[2] // patch_features
        ).to(latents)

    def pre_transition(
        self,
        latents: torch.Tensor,
        row: Batch,
        ctx: StepContext,
        transition: Transition,
    ) -> torch.Tensor:
        source = ConditionImage.parse(self.source).latents(row).to(latents)
        noise = _row_tensor(row, "noisy_latents").to(latents)
        if source.shape != latents.shape or noise.shape != latents.shape:
            raise ValueError(
                f"Differential diffusion requires source {self.source} and noisy_latents to match the current "
                f"latent shape {tuple(latents.shape)}; got {tuple(source.shape)} and {tuple(noise.shape)}."
            )
        sigma = latents.new_tensor(transition.sigma)
        reference = (1.0 - sigma) * source + sigma * noise
        keep_reference = (
            1.0 - self._edit_strength(row, latents)
        ) > ctx.item_index / ctx.num_items
        return torch.where(keep_reference, reference, latents)
