from dataclasses import dataclass
from typing import Literal

import torch

from ..plan import (
    EvalRequest,
    SamplingPlan,
    SolverRuntimeState,
    StepContext,
    Transition,
    TransitionGen,
    TransitionResult,
    zero_log_prob,
)
from .base import BaseSolver, solver_registry


@dataclass(frozen=True, slots=True)
class UniPCTransition(Transition):
    """UniPC transition with the plan-compiled predictor order cap."""

    order_cap: int = 1
    """``lower_order_final`` cap near the grid end; warmup happens at runtime."""


@dataclass(frozen=True, slots=True)
class UniPCRuntimeState(SolverRuntimeState):
    """Rolling multistep history (most recent last), rebuilt per transition."""

    x0_history: tuple[torch.Tensor, ...]
    """x0 predictions evaluated on pre-corrector samples."""
    sigma_history: tuple[float, ...]
    last_sample: torch.Tensor
    """The (corrected) sample the previous predictor stepped from."""
    prev_order: int
    """Predictor order used at the previous step; the corrector reuses it."""


@solver_registry.register("flow_unipc")
class FlowUniPCSolver(BaseSolver):
    """Deterministic UniPC multistep solver for flow matching (predict-x0, B(h)).

    Native port of HiDream-O1's ``FlowUniPCMultistepScheduler`` (the diffusers
    UniPC adapted to flow-matching sigmas) onto the plan-as-data interface:
    predict-x0 with the UniC corrector enabled, so the effective accuracy is
    ``order + 1``. Per reference defaults: ``order=2``, ``solver_type="bh2"``,
    ``lower_order_final=True``. Works with a zero-terminal sigma grid (the final
    step collapses to returning x0 exactly).

    Deterministic: usable for inference and for NFT/AWM/RAM-style rollouts, but
    not for GRPO step log-prob replay.
    """

    type: Literal["flow_unipc"] = "flow_unipc"
    order: Literal[1, 2] = 2
    solver_type: Literal["bh1", "bh2"] = "bh2"
    use_corrector: bool = True
    lower_order_final: bool = True

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        num_steps = len(sigmas) - 1
        return [
            UniPCTransition(
                solver=self,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=0.0,
                order_cap=(
                    min(self.order, num_steps - index)
                    if self.lower_order_final
                    else self.order
                ),
            )
            for index, (sigma, sigma_next) in enumerate(
                zip(sigmas[:-1], sigmas[1:], strict=True)
            )
        ]

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        assert isinstance(tr, UniPCTransition)
        state = ctx.solver_state
        assert state is None or isinstance(state, UniPCRuntimeState)

        out = yield EvalRequest(latents=ctx.latents, sigma=tr.sigma)
        latents = ctx.latents
        sigma_t = latents.new_tensor(tr.sigma)
        x0 = self._velocity_to_x0(out.velocity, latents, sigma_t)

        # Reference semantics: the corrector for the previous step runs first,
        # before the history shift, reusing the previous predictor order and
        # this transition's fresh model evaluation.
        sample = latents
        if self.use_corrector and state is not None:
            sample = self._multistep_uni_c_bh_update(
                this_x0=x0,
                sigma=sigma_t,
                x0_history=state.x0_history,
                sigma_history=tuple(
                    latents.new_tensor(value) for value in state.sigma_history
                ),
                last_sample=state.last_sample,
                order=state.prev_order,
            )

        x0_history = (*(state.x0_history if state is not None else ()), x0)
        sigma_history = (*(state.sigma_history if state is not None else ()), tr.sigma)
        # Multistep warmup from the runtime history length, so a sliced plan
        # restarts cleanly from an empty history.
        this_order = min(tr.order_cap, len(x0_history))

        next_latents = self._multistep_uni_p_bh_update(
            sample=sample,
            sigma=sigma_t,
            sigma_next=latents.new_tensor(tr.sigma_next),
            x0_history=x0_history,
            sigma_history=tuple(latents.new_tensor(value) for value in sigma_history),
            order=this_order,
        )

        next_state = UniPCRuntimeState(
            x0_history=x0_history[-self.order :],
            sigma_history=sigma_history[-self.order :],
            last_sample=sample,
            prev_order=this_order,
        )
        recorded = (
            self._make_recorded_step(tr, ctx, next_latents, zero_log_prob(latents))
            if tr.record
            else None
        )
        return TransitionResult(
            next_latents=next_latents,
            recorded=recorded,
            next_solver_state=next_state,
        )

    @staticmethod
    def _log_snr(sigma: torch.Tensor) -> torch.Tensor:
        alpha_t, sigma_t = BaseSolver._sigma_to_alpha_sigma_t(sigma)
        return torch.log(alpha_t) - torch.log(sigma_t)

    def _b_h(self, hh: torch.Tensor) -> torch.Tensor:
        return torch.expm1(hh) if self.solver_type == "bh2" else hh

    def _multistep_uni_p_bh_update(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        x0_history: tuple[torch.Tensor, ...],
        sigma_history: tuple[torch.Tensor, ...],
        order: int,
    ) -> torch.Tensor:
        m0 = x0_history[-1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_next)
        lambda_s0 = self._log_snr(sigma)
        h = self._log_snr(sigma_next) - lambda_s0

        d1s: list[torch.Tensor] = []
        for i in range(1, order):
            rk = (self._log_snr(sigma_history[-(i + 1)]) - lambda_s0) / h
            d1s.append((x0_history[-(i + 1)] - m0) / rk)

        hh = -h  # predict_x0
        h_phi_1 = torch.expm1(hh)

        x_t = sigma_next / sigma * sample - alpha_t * h_phi_1 * m0
        if d1s:
            # order == 2: the reference uses the simplified rhos_p = [0.5].
            x_t = x_t - alpha_t * self._b_h(hh) * (0.5 * d1s[0])
        return x_t

    def _multistep_uni_c_bh_update(
        self,
        this_x0: torch.Tensor,
        sigma: torch.Tensor,
        x0_history: tuple[torch.Tensor, ...],
        sigma_history: tuple[torch.Tensor, ...],
        last_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """Correct the current sample using the fresh model output.

        Reference semantics: runs *before* the history shift with the previous
        step's predictor order, stepping again from ``last_sample`` over
        ``[sigma_history[-1], sigma]`` with the extra
        ``D1_t = x0(z_i) - x0(z_{i-1})`` difference term.
        """
        m0 = x0_history[-1]
        x = last_sample
        sigma_s0 = sigma_history[-1]

        alpha_t, _ = self._sigma_to_alpha_sigma_t(sigma)
        lambda_s0 = self._log_snr(sigma_s0)
        h = self._log_snr(sigma) - lambda_s0

        rks: list[torch.Tensor] = []
        d1s: list[torch.Tensor] = []
        for i in range(1, order):
            rk = (self._log_snr(sigma_history[-(i + 1)]) - lambda_s0) / h
            rks.append(rk)
            d1s.append((x0_history[-(i + 1)] - m0) / rk)

        hh = -h  # predict_x0
        h_phi_1 = torch.expm1(hh)
        b_h = self._b_h(hh)

        if order == 1:
            corr_res = torch.zeros_like(m0)
            rho_last = torch.tensor(0.5, device=m0.device, dtype=m0.dtype)
        else:
            rks_t = torch.stack([*rks, torch.ones_like(h)]).reshape(-1)
            r_mat = torch.stack([rks_t ** (i - 1) for i in range(1, order + 1)])
            b: list[torch.Tensor] = []
            h_phi_k = h_phi_1 / hh - 1
            factorial_i = 1
            for i in range(1, order + 1):
                b.append(h_phi_k * factorial_i / b_h)
                factorial_i *= i + 1
                h_phi_k = h_phi_k / hh - 1 / factorial_i
            rhos_c = torch.linalg.solve(r_mat, torch.stack(b).reshape(-1))
            corr_res = torch.zeros_like(m0)
            for k in range(order - 1):
                corr_res = corr_res + rhos_c[k] * d1s[k]
            rho_last = rhos_c[-1]

        d1_t = this_x0 - m0
        return (
            sigma / sigma_s0 * x
            - alpha_t * h_phi_1 * m0
            - alpha_t * b_h * (corr_res + rho_last * d1_t)
        )
