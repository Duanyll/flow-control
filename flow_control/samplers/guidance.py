"""Guidance axis (axis 2 of the plan-as-data design): sampling middleware.

Branch *evaluation* (cond/uncond batched forwards + FSDP alignment) lives in
``flow_control/samplers/executor.py``. Guidance can prepare the latent once at
each transition boundary, then combine the evaluated branches. Guidance configs
are pydantic registry members (same mechanics as ``solver_registry``); their
live per-run state is a plain :class:`GuidanceState` carried by ``StepContext``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import torch
from einops import repeat
from pydantic import BaseModel, ConfigDict, Field

from flow_control.utils.registry import Registry, RegistryUnion

from .plan import BranchEvals, GuidanceOutput, GuidanceState, StepContext, Transition

if TYPE_CHECKING:
    from .executor import Run


class BaseGuidance(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    def init_state(self) -> GuidanceState | None:
        """Fresh per-run state, seeded into ``StepContext`` at run start."""
        return None

    def prepare_transition(
        self,
        run: Run,
        item_index: int,
    ) -> tuple[torch.Tensor, GuidanceState | None]:
        """Prepare a run's latent before one plan item starts.

        Called exactly once per plan item, before its transition generator can
        request any model evaluations. The default is a no-op. Return state as
        a new immutable value rather than mutating the current state in place.
        Do not mutate the current latent tensor in place.
        """
        return run.ctx.latents, run.ctx.guidance_state

    @abstractmethod
    def needs_negative(self) -> bool:
        """Whether this guidance wants an unconditional branch eval."""

    @abstractmethod
    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        """Combine branch velocities into one guided velocity.

        Called once per eval; the returned state takes effect immediately
        (multi-eval transitions advance it at every intermediate eval). Treat
        ``state`` and any tensors it contains as immutable: recorded steps may
        retain the same object as a transition-boundary snapshot.
        """


guidance_registry: Registry[BaseGuidance] = Registry("guidance", base=BaseGuidance)


@guidance_registry.register("cfg")
class ClassifierFreeGuidance(BaseGuidance):
    """Classifier-free guidance with optional norm-preserving rescale."""

    type: Literal["cfg"] = "cfg"
    scale: float = 1.0
    """
    TRUE classifier-free guidance scale. For guidance distilled models like FLUX, true
    CFG should not be applied and this should be kept at 1.0. Set their guidance
    embeddings value in ModelAdapter instead.
    """
    renorm: bool = False
    renorm_eps: float = 1e-8
    renorm_min: float = 0.0

    def needs_negative(self) -> bool:
        return self.scale > 1.0

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        if evals.uncond is None:
            return GuidanceOutput(velocity=evals.cond, branches=evals), state

        combined = evals.uncond + (evals.cond - evals.uncond) * self.scale
        if self.renorm:
            cond_norm = torch.norm(evals.cond, dim=2, keepdim=True)
            noise_norm = torch.norm(combined, dim=2, keepdim=True)
            combined = combined * (cond_norm / (noise_norm + self.renorm_eps)).clamp(
                min=self.renorm_min, max=1.0
            )
        return GuidanceOutput(velocity=combined, branches=evals), state


@dataclass(frozen=True, slots=True)
class MomentumGuidanceState(GuidanceState):
    """Exponentially smoothed velocity carried between guidance evals."""

    momentum: torch.Tensor | None = None


@guidance_registry.register("momentum")
class MomentumGuidance(ClassifierFreeGuidance):
    """CFG followed by the legacy velocity-momentum extrapolation.

    This is the functional counterpart of the former ``MomentumGuidedSampler``:
    runtime momentum lives in each run's immutable guidance state rather than
    on the shared sampler configuration.
    """

    type: Literal["momentum"] = "momentum"
    alpha: float
    beta: float

    def init_state(self) -> GuidanceState | None:
        return MomentumGuidanceState()

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        if not isinstance(state, MomentumGuidanceState):
            raise TypeError("MomentumGuidance requires MomentumGuidanceState.")
        cfg_output, _ = super().combine(evals, ctx, state)
        velocity = cfg_output.velocity
        momentum = velocity if state.momentum is None else state.momentum
        guided_velocity = velocity + self.alpha * (velocity - momentum)
        next_momentum = (1.0 - self.beta) * velocity + self.beta * momentum
        return (
            GuidanceOutput(velocity=guided_velocity, branches=evals),
            MomentumGuidanceState(momentum=next_momentum),
        )


Guidance = Annotated[
    BaseGuidance,
    # A bare number is the CFG scale: ``"guidance": 4.5``.
    RegistryUnion(guidance_registry, "type", number_as=("cfg", "scale")),
]


def _batch_tensor(run: Run, field: str) -> torch.Tensor:
    batch = cast("dict[str, Any]", run.batch)
    value = batch.get(field)
    if not isinstance(value, torch.Tensor):
        raise KeyError(
            f"Differential diffusion requires tensor field {field!r}; "
            f"available batch fields: {sorted(batch)}."
        )
    return value


@guidance_registry.register("differential")
class DifferentialDiffusionGuidance(BaseGuidance):
    """Differential Diffusion over the standard inpaint batch fields.

    ``inpaint_mask_latents`` is an edit-strength map: white pixels are released
    at the start of denoising, black pixels follow the noised
    ``inpaint_latents`` reference trajectory through the final transition, and
    gray values choose intermediate release times. ``InpaintProcessor`` packs
    the mask with the same patch geometry as the model latents.
    """

    type: Literal["differential"] = "differential"
    inner: Guidance = Field(default_factory=ClassifierFreeGuidance)

    def init_state(self) -> GuidanceState | None:
        return self.inner.init_state()

    def needs_negative(self) -> bool:
        return self.inner.needs_negative()

    @staticmethod
    def _edit_strength(run: Run, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError(
                "Differential diffusion requires packed BND latents, got "
                f"shape {tuple(latents.shape)}."
            )

        mask = _batch_tensor(run, "inpaint_mask_latents")
        if mask.ndim != 3:
            raise ValueError(
                "Differential diffusion expects inpaint_mask_latents in BND "
                f"format, got shape {tuple(mask.shape)}."
            )
        if mask.shape[:2] != latents.shape[:2]:
            raise ValueError(
                "Differential diffusion requires inpaint_mask_latents to match "
                "the latent batch and token dimensions; got "
                f"{tuple(mask.shape)} and {tuple(latents.shape)}."
            )
        patch_features = mask.shape[2]
        if patch_features == 0 or latents.shape[2] % patch_features != 0:
            raise ValueError(
                "Differential diffusion cannot expand packed mask features "
                f"{patch_features} over latent features {latents.shape[2]}."
            )
        return repeat(
            mask,
            "b n p -> b n (c p)",
            c=latents.shape[2] // patch_features,
        ).to(
            device=latents.device,
            dtype=latents.dtype,
        )

    def prepare_transition(
        self,
        run: Run,
        item_index: int,
    ) -> tuple[torch.Tensor, GuidanceState | None]:
        # Decorator order is inside-out: preserve any latent preparation and
        # state transition performed by the wrapped guidance.
        latents, state = self.inner.prepare_transition(run, item_index)
        item = run.plan[item_index]
        if not isinstance(item, Transition):
            return latents, state

        if item.sigma_next > item.sigma:
            raise ValueError(
                "Differential diffusion only supports denoising transitions, "
                f"got sigma {item.sigma} -> {item.sigma_next}."
            )

        source = _batch_tensor(run, "inpaint_latents").to(
            device=latents.device, dtype=latents.dtype
        )
        noise = _batch_tensor(run, "noisy_latents").to(
            device=latents.device, dtype=latents.dtype
        )
        if source.shape != latents.shape or noise.shape != latents.shape:
            raise ValueError(
                "Differential diffusion requires inpaint_latents and "
                "noisy_latents to match the current latent shape "
                f"{tuple(latents.shape)}; got {tuple(source.shape)} and "
                f"{tuple(noise.shape)}."
            )

        sigma = latents.new_tensor(item.sigma)
        reference = (1.0 - sigma) * source + sigma * noise
        edit_strength = self._edit_strength(run, latents)
        denoise_index = sum(
            isinstance(candidate, Transition) for candidate in run.plan[:item_index]
        )
        num_denoise_steps = sum(
            isinstance(candidate, Transition) for candidate in run.plan
        )
        progress = denoise_index / num_denoise_steps
        keep_reference = (1.0 - edit_strength) > progress
        return torch.where(keep_reference, reference, latents), state

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        return self.inner.combine(evals, ctx, state)


if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print

    adapter: TypeAdapter[BaseGuidance] = TypeAdapter(Guidance)
    guidance = adapter.validate_python({"type": "cfg", "scale": 4.5, "renorm": True})
    assert isinstance(guidance, ClassifierFreeGuidance)
    assert guidance.needs_negative()
    assert adapter.validate_json(adapter.dump_json(guidance)) == guidance
    # Bare-number shorthand: the CFG scale.
    assert adapter.validate_python(4.5) == ClassifierFreeGuidance(scale=4.5)
    differential = adapter.validate_python(
        {"type": "differential", "inner": {"type": "cfg", "scale": 4.5}}
    )
    assert isinstance(differential, DifferentialDiffusionGuidance)
    assert isinstance(differential.inner, ClassifierFreeGuidance)

    cond = torch.full((1, 2, 3), 3.0)
    uncond = torch.full((1, 2, 3), 1.0)
    latents = torch.zeros(1, 2, 3)
    ctx = StepContext(
        latents=latents, generator=None, solver_state=None, guidance_state=None
    )
    evals = BranchEvals(cond=cond, uncond=uncond, latents=latents, sigma=1.0)
    output, state = ClassifierFreeGuidance(scale=2.0).combine(evals, ctx, None)
    torch.testing.assert_close(output.velocity, uncond + (cond - uncond) * 2.0)
    assert output.branches is evals and state is None

    cond_only = BranchEvals(cond=cond, uncond=None, latents=latents, sigma=1.0)
    output, _ = guidance.combine(cond_only, ctx, None)
    assert output.velocity is cond

    print("[green]guidance smoke test passed[/green]")
