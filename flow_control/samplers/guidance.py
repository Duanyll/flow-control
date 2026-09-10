"""Named model branches and velocity combinations, with per-run state."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from flow_control.utils.registry import Registry, RegistryUnion

from .plan import BranchEvals, GuidanceOutput, GuidanceState, StepContext


@dataclass(frozen=True, slots=True)
class BranchSpec:
    name: str
    batch_key: str = "positive"
    variant: str | None = None
    """None preserves the current weights; 'base' disables LoRA."""
    optional: bool = False
    """A missing optional condition uses a discarded distributed dummy."""


Variant = str | Annotated[list[str | None], Field(min_length=1)] | None


def _variant_at(variant: Variant, item_index: int) -> str | None:
    return (
        variant[min(item_index, len(variant) - 1)]
        if isinstance(variant, list)
        else variant
    )


class BaseGuidance(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    def init_state(self) -> GuidanceState | None:
        return None

    @abstractmethod
    def branches(self, item_index: int) -> list[BranchSpec]:
        """Describe this step's model calls; independent of rank-local data."""

    def requires_negative(self, num_items: int) -> bool:
        return any(
            branch.batch_key == "negative"
            for index in range(num_items)
            for branch in self.branches(index)
        )

    @abstractmethod
    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        """Return velocity and new per-run state; called once per model eval."""


guidance_registry: Registry[BaseGuidance] = Registry("guidance", base=BaseGuidance)


@guidance_registry.register("cfg")
class ClassifierFreeGuidance(BaseGuidance):
    """CFG, optionally comparing different conditions or loaded LoRA weights.

    Variant lists select weights by executed step, holding the last entry after
    the list ends. ``negative_condition='positive'`` compares two weight variants
    on the same condition (constant-lambda signed guidance).
    """

    type: Literal["cfg"] = "cfg"
    scale: float = 1.0
    renorm: bool = False
    renorm_eps: float = 1e-8
    renorm_min: float = 0.0
    positive_variant: Variant = None
    negative_variant: Variant = None
    negative_condition: Literal["positive", "negative"] = "negative"

    def negative_branch(self, item_index: int) -> BranchSpec:
        return BranchSpec(
            "uncond",
            self.negative_condition,
            _variant_at(self.negative_variant, item_index),
            optional=True,
        )

    def branches(self, item_index: int) -> list[BranchSpec]:
        branches = [
            BranchSpec("cond", variant=_variant_at(self.positive_variant, item_index))
        ]
        if self.scale > 1.0:
            branches.append(self.negative_branch(item_index))
        return branches

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        cond = evals.velocities["cond"]
        uncond = evals.velocities.get("uncond")
        if uncond is None:
            return GuidanceOutput(velocity=cond), state
        combined = uncond + (cond - uncond) * self.scale
        if self.renorm:
            cond_norm = torch.norm(cond, dim=2, keepdim=True)
            noise_norm = torch.norm(combined, dim=2, keepdim=True)
            combined = combined * (cond_norm / (noise_norm + self.renorm_eps)).clamp(
                min=self.renorm_min, max=1.0
            )
        return GuidanceOutput(velocity=combined), state


Guidance = Annotated[
    BaseGuidance,
    RegistryUnion(guidance_registry, "type", number_as=("cfg", "scale")),
]


@guidance_registry.register("cfg_pp")
class CfgPlusPlusGuidance(BaseGuidance):
    """First-order CFG++: guided x0, unconditional re-noising.

    The effective velocity preserves the underlying solver's mean and noise
    formula. Flow and stochastic DDIM have different conversion coefficients.
    """

    type: Literal["cfg_pp"] = "cfg_pp"
    inner: Guidance = Field(default_factory=ClassifierFreeGuidance)

    def init_state(self) -> GuidanceState | None:
        return self.inner.init_state()

    def branches(self, item_index: int) -> list[BranchSpec]:
        branches = self.inner.branches(item_index)
        if not any(branch.name == "uncond" for branch in branches):
            branches.append(
                self.inner.negative_branch(item_index)
                if isinstance(self.inner, ClassifierFreeGuidance)
                else BranchSpec("uncond", "negative")
            )
        return [
            replace(branch, optional=False) if branch.name == "uncond" else branch
            for branch in branches
        ]

    @staticmethod
    def _kappa(evals: BranchEvals) -> float:
        from .solver.ddim import DDIMSolver
        from .solver.flow import FlowSolver

        sigma, sigma_next = evals.sigma, evals.sigma_next
        if sigma_next is None or not isinstance(evals.solver, (FlowSolver, DDIMSolver)):
            raise ValueError(
                "CFG++ requires a FlowSolver or DDIMSolver transition with sigma_next."
            )
        if not 0 <= sigma_next < sigma <= 1:
            raise ValueError(
                f"CFG++ requires 0 <= sigma_next < sigma <= 1, got {sigma} -> {sigma_next}."
            )
        if sigma == 1.0:
            return 1.0
        numerator = sigma * (1 - sigma_next)
        if isinstance(evals.solver, FlowSolver):
            denominator = (1 + evals.eta**2 / 2) * (sigma - sigma_next)
        else:
            noise_squared = (evals.eta * sigma_next) ** 2 * max(
                1 - sigma_next**2 * (1 - sigma) ** 2 / numerator**2, 0.0
            )
            sigma_down = math.sqrt(max(sigma_next**2 - noise_squared, 0.0))
            denominator = numerator - (1 - sigma) * sigma_down
        return numerator / denominator

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        kappa = self._kappa(evals)
        uncond = evals.velocities.get("uncond")
        if uncond is None:
            raise ValueError(
                "CFG++ requires an unconditional branch, including at CFG scale <= 1."
            )
        output, state = self.inner.combine(evals, ctx, state)
        return GuidanceOutput(
            velocity=uncond + kappa * (output.velocity - uncond)
        ), state


if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print

    adapter = TypeAdapter(Guidance)
    guidance = adapter.validate_python({"type": "cfg_pp", "inner": 0.5})
    assert guidance.requires_negative(1)
    assert adapter.validate_json(adapter.dump_json(guidance)) == guidance
    scheduled = ClassifierFreeGuidance(positive_variant=["default", "base"])
    assert [scheduled.branches(i)[0].variant for i in range(3)] == [
        "default",
        "base",
        "base",
    ]
    print("[green]guidance configuration smoke passed[/green]")
