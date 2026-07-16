import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry, RegistryUnion


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


@dataclass(slots=True)
class FlashSolverState(SolverState):
    num_steps: int
    step_index: int = 0


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
        generator: torch.Generator | None = None,
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


solver_registry: Registry[BaseSolver] = Registry("solver", base=BaseSolver)


@solver_registry.register("flow")
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
        generator: torch.Generator | None = None,
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
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
            next_latents = mean + noise_scale * noise

        return StepResult(
            next_latents=next_latents,
            log_prob=self._normal_log_prob(next_latents, mean, noise_scale),
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )


@solver_registry.register("dance")
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
        generator: torch.Generator | None = None,
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
                noise = torch.randn(
                    latents.shape,
                    dtype=latents.dtype,
                    device=latents.device,
                    generator=generator,
                )
                next_latents = mean + std_dev_t * noise
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


@solver_registry.register("ddim")
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
        generator: torch.Generator | None = None,
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
            noise = torch.randn(
                sample.shape,
                dtype=sample.dtype,
                device=sample.device,
                generator=generator,
            )
            next_latents = mean + noise_scale * noise

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
        generator: torch.Generator | None = None,
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
            generator=generator,
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


@solver_registry.register("cps")
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
        generator: torch.Generator | None = None,
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
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
            next_latents = mean + std_dev_t * noise

        log_prob = -((next_latents.detach() - mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return StepResult(
            next_latents=next_latents,
            log_prob=log_prob,
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )


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


@solver_registry.register("flash")
class FlashSolver(BaseSolver):
    """HiDream-O1 Dev's "flash" sampler: full re-noise at every step.

    Unlike an SDE-Euler step (which perturbs the ODE step with partial noise),
    each step extracts x0 and jumps to ``sigma_next`` with completely fresh
    noise: ``z' = (1 - sigma') * x0 + sigma' * s_noise * clip(eps)``, where
    ``s_noise`` interpolates linearly from ``noise_scale_start`` to
    ``noise_scale_end`` over the run and the noise is clamped at
    ``noise_clip_std`` times its own empirical std. Defaults follow the official
    Dev pipeline (7.5 pixel-space noise scale / 8 latent scaling = 0.9375;
    clip 2.5).

    The transition is Gaussian (``mean = (1 - sigma') * x0``,
    ``std = sigma' * s_noise``), so step log-probs are available for GRPO-style
    replay; with clipping enabled the Gaussian log-prob is an approximation
    (~1.2% of mass clipped at 2.5 std). ``eta`` acts as a gate only (the
    sampler zeroes it outside the trajectory window): ``eta == 0`` or a zero
    ``sigma_next`` falls back to the deterministic Euler step, which coincides
    with the re-noise formula at ``sigma' = 0``.
    """

    type: Literal["flash"] = "flash"
    eta: float = 1.0
    noise_scale_start: float = 0.9375
    noise_scale_end: float = 0.9375
    noise_clip_std: float = 2.5

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    def init_state(self, num_steps: int) -> FlashSolverState:
        return FlashSolverState(num_steps=num_steps)

    def _noise_scale(self, state: FlashSolverState) -> float:
        if state.num_steps <= 1:
            return self.noise_scale_start
        frac = state.step_index / (state.num_steps - 1)
        return (
            self.noise_scale_start
            + (self.noise_scale_end - self.noise_scale_start) * frac
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
        if not isinstance(state, FlashSolverState):
            msg = f"{self.type} requires FlashSolverState."
            raise TypeError(msg)
        next_state = FlashSolverState(
            num_steps=state.num_steps, step_index=state.step_index + 1
        )

        step_eta = self.eta if eta is None else eta
        if step_eta == 0.0 or bool((sigma_next <= 0.0).all()):
            dt = sigma_next - sigma
            return self._deterministic_step(latents, velocity, dt, state=next_state)

        x0 = self._velocity_to_x0(velocity, latents, sigma)
        mean = (1.0 - sigma_next) * x0
        std_dev = sigma_next * self._noise_scale(state)

        next_latents = prev_sample
        if next_latents is None:
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
            if self.noise_clip_std > 0:
                clip_val = self.noise_clip_std * noise.std()
                noise = noise.clamp(min=-clip_val, max=clip_val)
            next_latents = mean + std_dev * noise

        return StepResult(
            next_latents=next_latents,
            log_prob=self._normal_log_prob(next_latents, mean, std_dev),
            mean=mean,
            std_dev=self._scalar_like(std_dev, latents),
            state=next_state,
        )


Solver = Annotated[BaseSolver, RegistryUnion(solver_registry, "type")]
