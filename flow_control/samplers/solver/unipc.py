from dataclasses import dataclass
from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry


@dataclass(slots=True)
class UniPCSolverState(SolverState):
    order: int
    num_steps: int
    step_index: int = 0
    lower_order_nums: int = 0
    this_order: int = 1
    """Predictor order used at the previous step; the corrector reuses it."""
    model_outputs: tuple[torch.Tensor | None, ...] = ()
    """x0 history (computed on pre-corrector samples), most recent last."""
    sigmas: tuple[torch.Tensor | None, ...] = ()
    last_sample: torch.Tensor | None = None
    """The (corrected) sample the previous predictor stepped from."""


@solver_registry.register("flow_unipc")
class FlowUniPCSolver(BaseSolver):
    """Deterministic UniPC multistep solver for flow matching (predict-x0, B(h)).

    Native port of HiDream-O1's ``FlowUniPCMultistepScheduler`` (the diffusers
    UniPC adapted to flow-matching sigmas) onto the stateless-step interface:
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

    @property
    def supports_step_log_prob(self) -> bool:
        return False

    def init_state(self, num_steps: int) -> UniPCSolverState:
        return UniPCSolverState(
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
        if not isinstance(state, UniPCSolverState):
            msg = f"{self.type} requires UniPCSolverState."
            raise TypeError(msg)

        x0 = self._velocity_to_x0(velocity, latents, sigma)

        sample = latents
        if (
            self.use_corrector
            and state.step_index > 0
            and state.last_sample is not None
        ):
            sample = self._multistep_uni_c_bh_update(
                this_x0=x0, sigma=sigma, state=state
            )

        model_outputs = (*state.model_outputs[1:], x0)
        sigma_history = (*state.sigmas[1:], sigma)

        if self.lower_order_final:
            this_order = min(self.order, state.num_steps - state.step_index)
        else:
            this_order = self.order
        this_order = min(this_order, state.lower_order_nums + 1)  # multistep warmup

        next_latents = self._multistep_uni_p_bh_update(
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            model_outputs=model_outputs,
            sigma_history=sigma_history,
            order=this_order,
        )

        next_state = UniPCSolverState(
            order=state.order,
            num_steps=state.num_steps,
            step_index=state.step_index + 1,
            lower_order_nums=min(state.lower_order_nums + 1, self.order),
            this_order=this_order,
            model_outputs=model_outputs,
            sigmas=sigma_history,
            last_sample=sample,
        )
        return StepResult(
            next_latents=next_latents,
            log_prob=self._zero_log_prob(latents),
            mean=next_latents,
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
        model_outputs: tuple[torch.Tensor | None, ...],
        sigma_history: tuple[torch.Tensor | None, ...],
        order: int,
    ) -> torch.Tensor:
        m0 = model_outputs[-1]
        assert m0 is not None
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_next)
        lambda_t = self._log_snr(sigma_next)
        lambda_s0 = self._log_snr(sigma)
        h = lambda_t - lambda_s0

        d1s: list[torch.Tensor] = []
        for i in range(1, order):
            sigma_si = sigma_history[-(i + 1)]
            mi = model_outputs[-(i + 1)]
            if sigma_si is None or mi is None:
                msg = "UniPC predictor is missing multistep history."
                raise ValueError(msg)
            rk = (self._log_snr(sigma_si) - lambda_s0) / h
            d1s.append((mi - m0) / rk)

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
        state: UniPCSolverState,
    ) -> torch.Tensor:
        """Correct the current sample using the fresh model output.

        Reference semantics: runs *before* the history shift with the previous
        step's predictor order, stepping again from ``last_sample`` over
        ``[sigmas[-1], sigma]`` with the extra ``D1_t = x0(z_i) - x0(z_{i-1})``
        difference term.
        """
        m0 = state.model_outputs[-1]
        x = state.last_sample
        sigma_s0 = state.sigmas[-1]
        order = state.this_order
        if m0 is None or x is None or sigma_s0 is None:
            msg = "UniPC corrector is missing multistep history."
            raise ValueError(msg)

        alpha_t, _ = self._sigma_to_alpha_sigma_t(sigma)
        lambda_t = self._log_snr(sigma)
        lambda_s0 = self._log_snr(sigma_s0)
        h = lambda_t - lambda_s0

        rks: list[torch.Tensor] = []
        d1s: list[torch.Tensor] = []
        for i in range(1, order):
            sigma_si = state.sigmas[-(i + 1)]
            mi = state.model_outputs[-(i + 1)]
            if sigma_si is None or mi is None:
                msg = "UniPC corrector is missing multistep history."
                raise ValueError(msg)
            rk = (self._log_snr(sigma_si) - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

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
