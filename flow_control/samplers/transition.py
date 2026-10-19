"""Transition nodes: one step that moves the row from its σ to sigma_next."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, cast

import torch
from einops import repeat
from pydantic import Field

from flow_control.utils.condition_image import ConditionImage, ConditionImageSpec
from flow_control.utils.registry import Registry, RegistryUnion

from .calls import Calls
from .node import BaseNode, Frame, Row, caller, derived, randn
from .prediction import Model, Prediction


@dataclass(slots=True)
class StepLogProb:
    log_prob: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor


def normal_log_prob(
    sample: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    sample, mean, scale = sample.float(), mean.float(), scale.float()
    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * scale**2)
        - torch.log(scale)
        - torch.log(
            torch.sqrt(
                torch.tensor(2 * math.pi, device=sample.device, dtype=sample.dtype)
            )
        )
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def euler_moments(
    x: torch.Tensor, velocity: torch.Tensor, sigma: float, sigma_next: float, eta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flow SDE-Euler moments: ``(mean, diffusion coefficient, noise std)``."""
    sigma_t = x.new_tensor(sigma)
    sigma_next_t = x.new_tensor(sigma_next)
    dt = sigma_next_t - sigma_t
    sigma_denom = torch.where(sigma_t == 1.0, sigma_next_t, sigma_t)
    std_dev_t = torch.sqrt(sigma_t / (1 - sigma_denom)) * eta
    mean = (
        x * (1 + std_dev_t**2 / (2 * sigma_t) * dt)
        + velocity * (1 + std_dev_t**2 * (1 - sigma_t) / (2 * sigma_t)) * dt
    )
    return mean, std_dev_t, std_dev_t * torch.sqrt(-dt)


def ddim_moments(
    x: torch.Tensor, velocity: torch.Tensor, sigma: float, sigma_next: float, eta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """DDIM update moments: ``(mean, noise std)``."""
    sigma_t = x.new_tensor(sigma)
    sigma_next_t = x.new_tensor(sigma_next)
    pred_original_sample = x - sigma_t * velocity
    std_dev_t = eta * sigma_next_t
    dt_sqrt = torch.sqrt(
        torch.clamp(
            1.0
            - sigma_next_t**2
            * (1 - sigma_t) ** 2
            / (sigma_t**2 * (1 - sigma_next_t) ** 2),
            min=0.0,
        )
    )
    noise_scale = std_dev_t * dt_sqrt
    noise_pred = (x - (1 - sigma_t) * pred_original_sample) / sigma_t
    mean = (1 - sigma_next_t) * pred_original_sample + torch.sqrt(
        torch.clamp(sigma_next_t**2 - noise_scale**2, min=0.0)
    ) * noise_pred
    return mean, noise_scale


class StepFrame(Frame):
    sigma: float
    sigma_next: float
    latents: torch.Tensor
    next_latents: torch.Tensor
    velocity: torch.Tensor | None
    """The step's last velocity; never stored on the row."""


class BaseTransition(BaseNode):
    frame_type = StepFrame

    def __call__(self, row: Row, sigma_next: float) -> Calls[None]:
        with self.enter(row) as step:
            step.sigma, step.sigma_next = row["sigma"], sigma_next
            step.latents, step.velocity = row["noisy_latents"], None
            row["sigma_next"] = sigma_next  # flow maps read the step's target
            try:
                yield from self.run(row, step)
            finally:
                row.pop("sigma_next", None)  # only inside a step
            assert row["sigma"] == sigma_next
            step.next_latents = row["noisy_latents"]

    def run(self, row: Row, step: StepFrame) -> Calls[None]:
        raise NotImplementedError

    def is_stochastic(self, step: StepFrame) -> bool:
        return False

    def log_prob(self, point: Row, velocity: torch.Tensor) -> StepLogProb:
        """The step's likelihood at ``point`` (x_t, next_latents, a StepFrame chain)."""
        raise NotImplementedError(f"GRPO has no step likelihood for {self.type!r}.")


transition_registry: Registry[BaseTransition] = Registry(
    "transition", base=BaseTransition
)
Transition = Annotated[BaseTransition, RegistryUnion(transition_registry, "type")]


class GaussianTransition(BaseTransition):
    """One prediction, then a step whose stochastic form is Gaussian."""

    eta: float = 0.0
    prediction: Prediction = Model()

    def is_stochastic(self, step: StepFrame) -> bool:
        return self.eta != 0.0 and 0.0 < step.sigma_next < step.sigma

    def run(self, row: Row, step: StepFrame) -> Calls[None]:
        step.velocity = yield from self.prediction(row)
        x = row["noisy_latents"]
        if self.is_stochastic(step):
            mean, scale, _ = self.moments(row, step, x, step.velocity)
            row["noisy_latents"] = mean + scale * self.noise(row)
        else:
            row["noisy_latents"] = self.deterministic(x, step.velocity, step)
        row["sigma"] = step.sigma_next

    def log_prob(self, point: Row, velocity: torch.Tensor) -> StepLogProb:
        step = cast(StepFrame, point["_frame"])
        x, velocity = point["noisy_latents"].float(), velocity.float()
        if not self.is_stochastic(step):
            mean = self.deterministic(x, velocity, step)
            return StepLogProb(
                torch.zeros(x.shape[0], device=x.device), mean, x.new_tensor(0.0)
            )
        mean, scale, std_dev = self.moments(point, step, x, velocity)
        return StepLogProb(
            self.density(point["next_latents"], mean, scale), mean, std_dev
        )

    def moments(
        self, row: Row, step: StepFrame, x: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(mean, noise scale, std_dev)``; Euler's std_dev is the diffusion coefficient GRPO's KL uses."""
        raise NotImplementedError

    def deterministic(
        self, x: torch.Tensor, velocity: torch.Tensor, step: StepFrame
    ) -> torch.Tensor:
        return x + velocity * (x.new_tensor(step.sigma_next) - x.new_tensor(step.sigma))

    def noise(self, row: Row) -> torch.Tensor:
        return randn(row)

    def density(
        self, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return normal_log_prob(sample, mean, std)


@transition_registry.register("euler")
class Euler(GaussianTransition):
    type: Literal["euler"] = "euler"

    def moments(
        self, row: Row, step: StepFrame, x: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std_dev, scale = euler_moments(
            x, velocity, step.sigma, step.sigma_next, self.eta
        )
        return mean, scale, std_dev


@transition_registry.register("ddim")
class Ddim(GaussianTransition):
    type: Literal["ddim"] = "ddim"

    def moments(
        self, row: Row, step: StepFrame, x: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = ddim_moments(x, velocity, step.sigma, step.sigma_next, self.eta)
        return mean, std, std

    def deterministic(
        self, x: torch.Tensor, velocity: torch.Tensor, step: StepFrame
    ) -> torch.Tensor:
        return ddim_moments(x, velocity, step.sigma, step.sigma_next, 0.0)[0]


@transition_registry.register("cps")
class Cps(GaussianTransition):
    """CPS step; its likelihood is the unnormalized residual objective."""

    type: Literal["cps"] = "cps"

    def moments(
        self, row: Row, step: StepFrame, x: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_t = x.new_tensor(step.sigma)
        sigma_next_t = x.new_tensor(step.sigma_next)
        std_dev_t = sigma_next_t * math.sin(self.eta * math.pi / 2)
        pred_original_sample = x - sigma_t * velocity
        noise_estimate = x + velocity * (1 - sigma_t)
        mean = pred_original_sample * (1 - sigma_next_t) + noise_estimate * torch.sqrt(
            sigma_next_t**2 - std_dev_t**2
        )
        return mean, std_dev_t, std_dev_t

    def density(
        self, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        residual = -((sample.detach().float() - mean) ** 2)
        return residual.mean(dim=tuple(range(1, residual.ndim)))


@transition_registry.register("dance")
class Dance(GaussianTransition):
    type: Literal["dance"] = "dance"

    def moments(
        self, row: Row, step: StepFrame, x: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_t = x.new_tensor(step.sigma)
        sigma_next_t = x.new_tensor(step.sigma_next)
        dsigma = sigma_next_t - sigma_t
        mean = x + dsigma * velocity
        pred_original_sample = x - sigma_t * velocity
        delta_t = sigma_t - sigma_next_t
        std_dev_t = self.eta * torch.sqrt(delta_t)
        score_estimate = -(x - pred_original_sample * (1 - sigma_t)) / sigma_t**2
        mean = mean - 0.5 * self.eta**2 * score_estimate * dsigma
        return mean, std_dev_t, std_dev_t


@transition_registry.register("flash")
class Flash(GaussianTransition):
    """HiDream-O1 Dev's full re-noise step: x0, then fresh clipped noise whose scale ramps over the grid."""

    type: Literal["flash"] = "flash"
    eta: float = 1.0
    noise_scale_start: float = 0.9375
    noise_scale_end: float = 0.9375
    noise_clip_std: float = 2.5

    def moments(
        self, row: Row, step: StepFrame, x: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .trajectory import BaseGrid, GridFrame

        grid = cast(
            GridFrame,
            caller(row, BaseGrid, who="flash ramps its noise scale over the grid"),
        )
        steps = len(grid.sigmas) - 1
        frac = grid.index / (steps - 1) if steps > 1 else 0.0
        scale = (
            self.noise_scale_start
            + (self.noise_scale_end - self.noise_scale_start) * frac
        )
        sigma_next_t = x.new_tensor(step.sigma_next)
        mean = (1.0 - sigma_next_t) * (x - x.new_tensor(step.sigma) * velocity)
        return mean, sigma_next_t * scale, sigma_next_t * scale

    def noise(self, row: Row) -> torch.Tensor:
        noise = randn(row)
        if self.noise_clip_std > 0:
            clip_val = self.noise_clip_std * noise.std()
            noise = noise.clamp(min=-clip_val, max=clip_val)
        return noise


@transition_registry.register("differential_diffusion")
class DifferentialDiffusion(BaseTransition):
    """Keep masked pixels on the noised reference until their edit strength releases them, then step."""

    type: Literal["differential_diffusion"] = "differential_diffusion"
    source: ConditionImageSpec = Field("inpaint", alias="from")
    transition: Transition = Euler()

    def run(self, row: Row, step: StepFrame) -> Calls[None]:
        from .trajectory import BaseGrid, GridFrame

        grid = cast(
            GridFrame,
            caller(
                row,
                BaseGrid,
                who="differential_diffusion releases pixels over the grid",
            ),
        )
        x = row["noisy_latents"]
        # The initial noise, kept from the first step: the grid starts at σ = 1.
        noise = derived(row, "_noise", lambda: x)
        source = ConditionImage.parse(self.source).latents(row).to(x)
        mask = row["inpaint_mask_latents"]
        strength = repeat(mask, "b n p -> b n (c p)", c=x.shape[2] // mask.shape[2]).to(
            x
        )
        sigma = x.new_tensor(step.sigma)
        reference = (1.0 - sigma) * source + sigma * noise
        keep = (1.0 - strength) > grid.index / (len(grid.sigmas) - 1)
        row["noisy_latents"] = torch.where(keep, reference, x)
        yield from self.transition(row, step.sigma_next)
