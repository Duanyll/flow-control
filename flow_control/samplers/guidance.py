"""CFG predictors; condition calls and their algebra live in the same component."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Annotated, ClassVar, Literal

import torch
from pydantic import Field

from flow_control.adapters.base import Batch
from flow_control.utils.logging import get_logger, warn_once

from .calls import Calls, gather
from .plan import EvalRequest, StepContext
from .prediction import Prediction, Predictor, WrappedPrediction, prediction_registry
from .tiling import TiledPrediction

logger = get_logger(__name__)
Variant = str | Annotated[list[str | None], Field(min_length=1)] | None


def _variant_at(variant: Variant, item_index: int) -> str | None:
    return (
        variant[min(item_index, len(variant) - 1)]
        if isinstance(variant, list)
        else variant
    )


@prediction_registry.register("cfg")
class ClassifierFreeGuidance(WrappedPrediction):
    """Predict each condition through an independent child, then guide.

    Inner state belongs to its condition branch. Wrapping this component in
    Momentum instead gives momentum of the combined prediction.
    """

    type: Literal["cfg"] = "cfg"
    inner: Prediction = Field(default_factory=TiledPrediction)
    scale: float = 1.0
    renorm: bool = False
    renorm_eps: float = 1e-8
    renorm_min: float = 0.0
    positive_variant: Variant = None
    negative_variant: Variant = None
    negative_condition: Literal["positive", "negative"] = "negative"
    negative_required: ClassVar[bool] = False

    def _needs_unconditional(self) -> bool:
        return self.scale > 1.0

    def requires_negative(self, num_items: int) -> bool:
        return (
            self._needs_unconditional() and self.negative_condition == "negative"
        ) or super().requires_negative(num_items)

    def variant_keys(self, num_items: int) -> list[str | None]:
        schedules = [self.positive_variant]
        if self._needs_unconditional():
            schedules.append(self.negative_variant)
        return list(
            dict.fromkeys(
                [
                    *(_variant_at(v, i) for i in range(num_items) for v in schedules),
                    *super().variant_keys(num_items),
                ]
            )
        )

    def bind(self, batch: Batch, negative_batch: Batch | None = None) -> Predictor:
        if self.inner.requires_negative(1) or any(
            isinstance(node, ClassifierFreeGuidance) for node in self.inner.walk()
        ):
            raise ValueError(
                f"{self.type} evaluates its inner predictor separately per condition; "
                "inner cannot request another negative condition. Place combined "
                "guidance outside this CFG, or put CFG inside Tiled instead."
            )
        positive = self.inner.bind(batch)
        negative_source = (
            batch if self.negative_condition == "positive" else negative_batch
        )
        negative = None
        if self._needs_unconditional():
            if negative_source is None:
                if self.negative_required:
                    raise ValueError(
                        "CFG++ requires a negative batch; enable the processor's negative conditioning."
                    )
                warn_once(
                    logger,
                    "CFG has no negative batch; falling back to conditional prediction.",
                )
            else:
                negative = self.inner.bind(negative_source)

        def predict(request: EvalRequest, ctx: StepContext) -> Calls[torch.Tensor]:
            calls = [
                positive(
                    replace(
                        request,
                        variant=_variant_at(self.positive_variant, ctx.item_index),
                    ),
                    ctx,
                )
            ]
            if negative is not None:
                calls.append(
                    negative(
                        replace(
                            request,
                            variant=_variant_at(self.negative_variant, ctx.item_index),
                        ),
                        ctx,
                    )
                )
            velocities = yield from gather(calls)
            if negative is None:
                return velocities[0]
            return self._guide(velocities[0], velocities[1], request)

        return predict

    def _guide(
        self, cond: torch.Tensor, uncond: torch.Tensor, request: EvalRequest
    ) -> torch.Tensor:
        """The shared CFG algebra; CFG++ additionally preserves its solver's mean."""
        combined = uncond + (cond - uncond) * self.scale
        if self.renorm:
            cond_norm = torch.norm(cond, dim=2, keepdim=True)
            noise_norm = torch.norm(combined, dim=2, keepdim=True)
            combined = combined * (cond_norm / (noise_norm + self.renorm_eps)).clamp(
                min=self.renorm_min, max=1.0
            )
        return combined


@prediction_registry.register("cfg_pp")
class CfgPlusPlusGuidance(ClassifierFreeGuidance):
    """First-order CFG++ with explicit CFG parameters and a single-condition child.

    Requires both conditions even at scale <= 1. Solver-dependent conversion
    stays here; it is only defined for Flow and DDIM transitions.
    """

    type: Literal["cfg_pp"] = "cfg_pp"
    negative_required: ClassVar[bool] = True

    def _needs_unconditional(self) -> bool:
        return True

    @staticmethod
    def _kappa(request: EvalRequest) -> float:
        from .solver.ddim import DDIMSolver
        from .solver.flow import FlowSolver

        sigma, sigma_next = float(request.sigma), request.sigma_next
        if sigma_next is None or not isinstance(
            request.solver, (FlowSolver, DDIMSolver)
        ):
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
        if isinstance(request.solver, FlowSolver):
            denominator = (1 + request.eta**2 / 2) * (sigma - sigma_next)
        else:
            noise_squared = (request.eta * sigma_next) ** 2 * max(
                1 - sigma_next**2 * (1 - sigma) ** 2 / numerator**2, 0.0
            )
            sigma_down = math.sqrt(max(sigma_next**2 - noise_squared, 0.0))
            denominator = numerator - (1 - sigma) * sigma_down
        return numerator / denominator

    def _guide(
        self, cond: torch.Tensor, uncond: torch.Tensor, request: EvalRequest
    ) -> torch.Tensor:
        kappa = self._kappa(request)
        guided = super()._guide(cond, uncond, request)
        return uncond + kappa * (guided - uncond)


if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print

    from flow_control.utils.registry import load_plugins

    from .sampler import Sampler

    def check_prediction_schema(schema: dict, expected: set[str]) -> None:
        containers = {
            "ClassifierFreeGuidance": "inner",
            "CfgPlusPlusGuidance": "inner",
            "TiledPrediction": "inner",
        }
        if "momentum" in expected:
            containers["MomentumGuidance"] = "inner"
        fields = [
            schema["$defs"][name]["properties"][field]
            for name, field in containers.items()
        ]
        fields.append(schema.get("properties", {}).get("guidance", schema))
        for field in fields:
            tagged = next(
                branch for branch in field["anyOf"] if "discriminator" in branch
            )
            assert set(tagged["discriminator"]["mapping"]) == expected

    adapter = TypeAdapter(Prediction)
    core_members = {"model", "tiled", "cfg", "cfg_pp"}
    check_prediction_schema(adapter.json_schema(), core_members)

    # The composition refactor initially cached Sampler during this plugin's
    # imports, omitting Momentum from both the root and recursive inner schemas.
    load_plugins(["flow_control.contrib.momentum_guidance"])
    check_prediction_schema(Sampler.model_json_schema(), core_members | {"momentum"})

    guidance = adapter.validate_python({"type": "cfg_pp", "scale": 0.5})
    assert guidance.requires_negative(1)
    assert adapter.validate_json(adapter.dump_json(guidance)) == guidance
    scheduled = ClassifierFreeGuidance(positive_variant=["default", "base"])
    assert scheduled.variant_keys(3) == ["default", "base"]
    print("[green]composed guidance configuration smoke passed[/green]")
