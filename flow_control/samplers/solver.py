import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, Tag


class SolverState:
    """Marker base class for per-solver runtime state."""


@dataclass(slots=True)
class StepResult:
    next_latents: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor
    state: SolverState | None = None


@dataclass(slots=True)
class DPMSolverState(SolverState):
    order: int
    num_steps: int
    step_index: int = 0
    lower_order_nums: int = 0
    model_outputs: tuple[torch.Tensor | None, ...] = ()
    sigmas: tuple[torch.Tensor | None, ...] = ()


class BaseSolver(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    eta: float = 0.0

    def init_state(self, num_steps: int) -> SolverState | None:
        return None

    @property
    def supports_step_log_prob(self) -> bool:
        return False

    def replay_step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor,
        state: SolverState | None = None,
    ) -> StepResult:
        return self.step(
            velocity=velocity,
            latents=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=prev_sample,
            eta=self.eta,
            state=state,
        )

    @abstractmethod
    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
    ) -> StepResult:
        raise NotImplementedError()

    @staticmethod
    def _deterministic_step(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        dt: torch.Tensor,
        state: SolverState | None = None,
    ) -> StepResult:
        next_latents = latents + velocity * dt
        return StepResult(
            next_latents=next_latents,
            log_prob=torch.zeros(latents.shape[0], device=latents.device),
            mean=next_latents,
            std_dev=torch.tensor(0.0, device=latents.device),
            state=state,
        )

    @staticmethod
    def _zero_log_prob(latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latents.shape[0], device=latents.device)

    @staticmethod
    def _scalar_like(value: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.squeeze() if value.ndim > 0 else value
        return torch.tensor(value, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _normal_log_prob(
        sample: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
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

    @staticmethod
    def _velocity_to_x0(
        velocity: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return sample - sigma * velocity

    @staticmethod
    def _sigma_to_alpha_sigma_t(
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma


class FlowSolver(BaseSolver):
    type: Literal["flow"] = "flow"

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
    ) -> StepResult:
        step_eta = self.eta if eta is None else eta
        dt = sigma_next - sigma
        if step_eta == 0.0:
            return self._deterministic_step(latents, velocity, dt, state=state)

        sigma_denom = torch.where(sigma == 1.0, sigma_next, sigma)
        std_dev_t = torch.sqrt(sigma / (1 - sigma_denom)) * step_eta
        mean = (
            latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + velocity * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        noise_scale = std_dev_t * torch.sqrt(-dt)

        next_latents = prev_sample
        if next_latents is None:
            next_latents = mean + noise_scale * torch.randn_like(latents)

        return StepResult(
            next_latents=next_latents,
            log_prob=self._normal_log_prob(next_latents, mean, noise_scale),
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )


class DanceSolver(BaseSolver):
    type: Literal["dance"] = "dance"

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
    ) -> StepResult:
        step_eta = self.eta if eta is None else eta
        dsigma = sigma_next - sigma
        mean = latents + dsigma * velocity

        pred_original_sample = latents - sigma * velocity
        if step_eta > 0.0:
            delta_t = sigma - sigma_next
            std_dev_t = step_eta * torch.sqrt(delta_t)
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
            mean = mean - 0.5 * step_eta**2 * score_estimate * dsigma
            next_latents = prev_sample
            if next_latents is None:
                next_latents = mean + std_dev_t * torch.randn_like(latents)
            log_prob = self._normal_log_prob(next_latents, mean, std_dev_t)
        else:
            std_dev_t = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
            next_latents = mean if prev_sample is None else prev_sample
            log_prob = self._zero_log_prob(latents)

        return StepResult(
            next_latents=next_latents,
            log_prob=log_prob,
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )


class DDIMSolver(BaseSolver):
    type: Literal["ddim"] = "ddim"

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    @staticmethod
    def _ddim_update(
        pred_original_sample: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        prev_sample: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std_dev_t = eta * sigma_next
        dt_sqrt = torch.sqrt(
            torch.clamp(
                1.0
                - sigma_next**2 * (1 - sigma) ** 2 / (sigma**2 * (1 - sigma_next) ** 2),
                min=0.0,
            )
        )
        noise_scale = std_dev_t * dt_sqrt
        noise_pred = (sample - (1 - sigma) * pred_original_sample) / sigma
        mean = (1 - sigma_next) * pred_original_sample + torch.sqrt(
            torch.clamp(sigma_next**2 - noise_scale**2, min=0.0)
        ) * noise_pred

        next_latents = prev_sample
        if next_latents is None:
            next_latents = mean + noise_scale * torch.randn_like(sample)

        return next_latents, mean, noise_scale

    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
    ) -> StepResult:
        step_eta = self.eta if eta is None else eta
        pred_original_sample = self._velocity_to_x0(velocity, latents, sigma)
        next_latents, mean, noise_scale = self._ddim_update(
            pred_original_sample=pred_original_sample,
            sample=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=step_eta,
            prev_sample=prev_sample,
        )

        if step_eta > 0.0:
            log_prob = self._normal_log_prob(next_latents, mean, noise_scale)
        else:
            log_prob = self._zero_log_prob(latents)

        return StepResult(
            next_latents=next_latents,
            log_prob=log_prob,
            mean=mean,
            std_dev=self._scalar_like(noise_scale, latents),
            state=state,
        )


class CPSSolver(BaseSolver):
    type: Literal["cps"] = "cps"

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
    ) -> StepResult:
        step_eta = self.eta if eta is None else eta
        dt = sigma_next - sigma
        if step_eta == 0.0:
            return self._deterministic_step(latents, velocity, dt, state=state)

        std_dev_t = sigma_next * math.sin(step_eta * math.pi / 2)
        pred_original_sample = latents - sigma * velocity
        noise_estimate = latents + velocity * (1 - sigma)
        mean = pred_original_sample * (1 - sigma_next) + noise_estimate * torch.sqrt(
            sigma_next**2 - std_dev_t**2
        )

        next_latents = prev_sample
        if next_latents is None:
            next_latents = mean + std_dev_t * torch.randn_like(latents)

        log_prob = -((next_latents.detach() - mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return StepResult(
            next_latents=next_latents,
            log_prob=log_prob,
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )


class DPMSolver(DDIMSolver):
    type: Literal["dpm"] = "dpm"
    order: Literal[1, 2]

    @property
    def supports_step_log_prob(self) -> bool:
        return False

    def init_state(self, num_steps: int) -> DPMSolverState:
        return DPMSolverState(
            order=self.order,
            num_steps=num_steps,
            model_outputs=tuple(None for _ in range(self.order)),
            sigmas=tuple(None for _ in range(self.order)),
        )

    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
    ) -> StepResult:
        if prev_sample is not None:
            msg = f"{self.type} is deterministic and does not support replay log-prob."
            raise ValueError(msg)
        if not isinstance(state, DPMSolverState):
            msg = f"{self.type} requires DPMSolverState."
            raise TypeError(msg)

        model_output = self._velocity_to_x0(velocity, latents, sigma)
        model_outputs = (*state.model_outputs[1:], model_output)
        sigma_history = (*state.sigmas[1:], sigma)

        lower_order_final = state.step_index == state.num_steps - 1
        lower_order_second = (
            state.step_index == state.num_steps - 2 and state.num_steps < 15
        )

        if self.order == 1 or state.lower_order_nums < 1 or lower_order_final:
            if state.step_index == 0 or lower_order_final:
                next_latents, mean, _ = self._ddim_update(
                    pred_original_sample=model_output,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=0.0,
                )
            else:
                next_latents = self._dpm_solver_first_order_update(
                    model_output=model_output,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                )
                mean = next_latents
        elif self.order == 2 or state.lower_order_nums < 2 or lower_order_second:
            next_latents = self._multistep_dpm_solver_second_order_update(
                model_outputs=model_outputs,
                sample=latents,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_prev=sigma_history[-2],
            )
            mean = next_latents
        else:
            msg = f"Unsupported DPM order: {self.order}"
            raise ValueError(msg)

        next_state = DPMSolverState(
            order=state.order,
            num_steps=state.num_steps,
            step_index=state.step_index + 1,
            lower_order_nums=min(state.lower_order_nums + 1, self.order),
            model_outputs=model_outputs,
            sigmas=sigma_history,
        )
        return StepResult(
            next_latents=next_latents,
            log_prob=self._zero_log_prob(latents),
            mean=mean,
            std_dev=torch.tensor(0.0, device=latents.device),
            state=next_state,
        )

    def replay_step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor,
        state: SolverState | None = None,
    ) -> StepResult:
        msg = f"{self.type} is deterministic and does not support replay log-prob."
        raise ValueError(msg)

    def _dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_next)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        return (sigma_t / sigma_s) * sample - (
            alpha_t * (torch.exp(-h) - 1.0)
        ) * model_output

    def _multistep_dpm_solver_second_order_update(
        self,
        model_outputs: tuple[torch.Tensor | None, ...],
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        if sigma_prev is None:
            msg = "DPM2 requires the previous model output in state."
            raise ValueError(msg)
        m0 = model_outputs[-1]
        m1 = model_outputs[-2]
        if m0 is None or m1 is None:
            msg = "DPM2 requires two cached model outputs."
            raise ValueError(msg)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_next)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_prev)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        d0 = m0
        d1 = (1.0 / r0) * (m0 - m1)

        return (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * d0
            - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1
        )


Solver = Annotated[
    Annotated[FlowSolver, Tag("flow")]
    | Annotated[DanceSolver, Tag("dance")]
    | Annotated[DDIMSolver, Tag("ddim")]
    | Annotated[CPSSolver, Tag("cps")]
    | Annotated[DPMSolver, Tag("dpm")],
    Discriminator("type"),
]
