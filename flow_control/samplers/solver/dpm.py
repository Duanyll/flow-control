from dataclasses import dataclass
from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry
from .ddim import DDIMSolver


@dataclass(slots=True)
class DPMSolverState(SolverState):
    order: int
    num_steps: int
    step_index: int = 0
    lower_order_nums: int = 0
    model_outputs: tuple[torch.Tensor | None, ...] = ()
    sigmas: tuple[torch.Tensor | None, ...] = ()


@solver_registry.register("dpm")
class DPMSolver(BaseSolver):
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
        generator: torch.Generator | None = None,
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
                next_latents, mean, _ = DDIMSolver._ddim_update(
                    pred_original_sample=model_output,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=0.0,
                    generator=generator,
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
