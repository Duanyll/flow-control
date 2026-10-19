"""Multistep grids: deterministic DPM++ / UniPC and the SA-Solver PEC loop."""

from __future__ import annotations

from typing import Literal

import torch

from .calls import Calls
from .node import Row, randn
from .prediction import Model, Prediction
from .trajectory import BaseGrid, GridFrame, trajectory_registry
from .transition import ddim_moments


def _lambda(sigma: torch.Tensor) -> torch.Tensor:
    return torch.log(1 - sigma) - torch.log(sigma)


@trajectory_registry.register("dpm")
class DpmGrid(BaseGrid):
    """Multistep DPM-Solver++ in x0-prediction form; DDIM on the first and last step."""

    type: Literal["dpm"] = "dpm"
    order: Literal[1, 2]
    prediction: Prediction = Model()

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.sigmas = self.sigmas(row)
        previous: tuple[torch.Tensor, float] | None = None  # (x0, sigma) last step
        for grid.index in range(self.steps):
            x = row["noisy_latents"]
            sigma, sigma_next = grid.sigmas[grid.index], grid.sigmas[grid.index + 1]
            velocity = yield from self.prediction(row)
            sigma_t = x.new_tensor(sigma)
            x0 = x - sigma_t * velocity
            if grid.index in (0, self.steps - 1):
                row["noisy_latents"] = ddim_moments(
                    x, velocity, sigma, sigma_next, 0.0
                )[0]
            elif self.order == 1:
                row["noisy_latents"] = _dpm_first_order(
                    x0, x, sigma_t, x.new_tensor(sigma_next)
                )
            else:
                assert previous is not None
                row["noisy_latents"] = _dpm_second_order(
                    x0,
                    previous[0],
                    x,
                    sigma_t,
                    x.new_tensor(sigma_next),
                    x.new_tensor(previous[1]),
                )
            row["sigma"] = sigma_next
            previous = (x0, sigma)

    def estimate_calls(self, row: Row) -> int:
        return self.steps * self.prediction.estimate_calls(row)


def _dpm_first_order(
    x0: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor
) -> torch.Tensor:
    alpha_t, sigma_t = 1 - sigma_next, sigma_next
    h = _lambda(sigma_next) - _lambda(sigma)
    return (sigma_t / sigma) * x - (alpha_t * (torch.exp(-h) - 1.0)) * x0


def _dpm_second_order(
    m0: torch.Tensor,
    m1: torch.Tensor,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    sigma_prev: torch.Tensor,
) -> torch.Tensor:
    alpha_t, sigma_t = 1 - sigma_next, sigma_next
    lambda_t, lambda_s0, lambda_s1 = (
        _lambda(sigma_next),
        _lambda(sigma),
        _lambda(sigma_prev),
    )
    h = lambda_t - lambda_s0
    r0 = (lambda_s0 - lambda_s1) / h
    d1 = (1.0 / r0) * (m0 - m1)
    return (
        (sigma_t / sigma) * x
        - (alpha_t * (torch.exp(-h) - 1.0)) * m0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1
    )


@trajectory_registry.register("sa")
class SaGrid(BaseGrid):
    """Second-order SA-Solver (data prediction, PEC) of the public AWM SD3.5 recipe.

    The grid head is capped at ``initial_time`` and the penultimate time halved
    like the reference; ``eta`` applies to target times inside the
    stochasticity band. The random PEC step has no step-wise density.
    """

    type: Literal["sa"] = "sa"
    eta: float = 0.4
    stochasticity_start: float = 0.3
    stochasticity_end: float = 0.9
    initial_time: float = 1.0 - 1e-3
    adjust_penultimate_time: bool = True
    prediction: Prediction = Model()

    def tau(self, t: float) -> float:
        return (
            self.eta if self.stochasticity_start <= t <= self.stochasticity_end else 0.0
        )

    def sigmas(self, row: Row) -> list[float]:
        times = super().sigmas(row)
        times[0] = min(times[0], self.initial_time)
        if self.adjust_penultimate_time:
            times[-2] = times[-3] / 2.0
        return times

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.sigmas = times = self.sigmas(row)
        row["sigma"] = times[0]
        x = row["noisy_latents"]
        velocity = yield from self.prediction(row)  # seeding eval at the current point
        models = [x - x.new_tensor(times[0]) * velocity]
        history = [times[0]]
        for grid.index in range(self.steps - 1):
            x, t_next = row["noisy_latents"], times[grid.index + 1]
            tau = self.tau(t_next)
            order: Literal[1, 2] = 1 if len(models) < 2 else 2
            noise = randn(row) if tau > 0.0 else torch.zeros_like(x)
            t, tau_t = x.new_tensor(t_next), x.new_tensor(tau)
            time_list = [x.new_tensor(value) for value in history]
            row["noisy_latents"] = predicted = sa_update(
                x, tau_t, models, time_list, noise, t, order, corrector=False
            )
            row["sigma"] = t_next
            velocity = yield from self.prediction(row)
            new_model = predicted - t * velocity
            if order == 2:
                row["noisy_latents"] = sa_update(
                    x,
                    tau_t,
                    [*models, new_model],
                    time_list,
                    noise,
                    t,
                    2,
                    corrector=True,
                )
            models, history = [*models, new_model][-2:], [*history, t_next][-2:]
        grid.index = self.steps - 1
        row["noisy_latents"], row["sigma"] = models[-1], times[-1]  # the final x0

    def estimate_calls(self, row: Row) -> int:
        return self.steps * self.prediction.estimate_calls(row)


def sa_update(
    x: torch.Tensor,
    tau: torch.Tensor,
    models: list[torch.Tensor],
    time_history: list[torch.Tensor],
    noise: torch.Tensor,
    t: torch.Tensor,
    order: Literal[1, 2],
    corrector: bool,
) -> torch.Tensor:
    """One Adams-Bashforth (predictor) or Adams-Moulton (corrector) SA-Solver update."""
    lambda_t, lambda_prev = _lambda(t), _lambda(time_history[-1])
    h = lambda_t - lambda_prev
    times = [*time_history, t] if corrector else time_history
    lambda_list = [_lambda(times[-(i + 1)]) for i in range(order)]
    scale = 1.0 + tau**2
    start, end = scale * lambda_prev, scale * lambda_t
    exp_delta = torch.exp(-(end - start))
    integral_0 = torch.exp(end) * (1.0 - exp_delta) / scale
    if order == 1:
        coefficients = [integral_0]
    else:
        integral_1 = (
            torch.exp(end) * ((end - 1.0) - (start - 1.0) * exp_delta) / scale**2
        )
        lambda_0, lambda_1 = lambda_list
        denominator = lambda_0 - lambda_1
        coefficients = [
            (integral_1 - lambda_1 * integral_0) / denominator,
            (-integral_1 + lambda_0 * integral_0) / denominator,
        ]
        # The few-step coefficient correction of the authors' implementation.
        if corrector:
            correction = torch.exp(scale * lambda_t) * (
                h / 2.0 - (scale * h - 1.0 + torch.exp(-scale * h)) / (scale**2 * h)
            )
        else:
            correction = (
                torch.exp(scale * lambda_t)
                * (h**2 / 2.0 - (scale * h - 1.0 + torch.exp(-scale * h)) / scale**2)
                / (lambda_prev - _lambda(time_history[-2]))
            )
        coefficients[0] = coefficients[0] + correction
        coefficients[1] = coefficients[1] - correction
    gradient = torch.zeros_like(x)
    factor = scale * t * torch.exp(-(tau**2) * lambda_t)
    for i, coefficient in enumerate(coefficients):
        gradient = gradient + factor * coefficient * models[-(i + 1)]
    noise_variance = 1.0 - torch.exp(-2.0 * tau**2 * h)
    noise_part = t * torch.sqrt(noise_variance.clamp_min(0.0)) * noise
    return torch.exp(-(tau**2) * h) * (t / time_history[-1]) * x + gradient + noise_part


@trajectory_registry.register("flow_unipc")
class UniPcGrid(BaseGrid):
    """Deterministic UniPC (predict-x0, B(h)) for flow matching, HiDream-O1's port.

    The UniC corrector runs before the history shift with the previous
    predictor order, so the effective accuracy is ``order + 1``.
    """

    type: Literal["flow_unipc"] = "flow_unipc"
    order: Literal[1, 2] = 2
    solver_type: Literal["bh1", "bh2"] = "bh2"
    use_corrector: bool = True
    lower_order_final: bool = True
    prediction: Prediction = Model()

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.sigmas = self.sigmas(row)
        x0s: list[torch.Tensor] = []  # x0 predictions on pre-corrector samples
        sigmas: list[float] = []
        last_sample, prev_order = None, 0
        for grid.index in range(self.steps):
            x = row["noisy_latents"]
            sigma, sigma_next = grid.sigmas[grid.index], grid.sigmas[grid.index + 1]
            velocity = yield from self.prediction(row)
            sigma_t = x.new_tensor(sigma)
            x0 = x - sigma_t * velocity
            sample = x
            if self.use_corrector and last_sample is not None:
                sample = self._corrector(
                    x0,
                    sigma_t,
                    x0s,
                    [x.new_tensor(v) for v in sigmas],
                    last_sample,
                    prev_order,
                )
            x0s, sigmas = [*x0s, x0], [*sigmas, sigma]
            order_cap = (
                min(self.order, self.steps - grid.index)
                if self.lower_order_final
                else self.order
            )
            this_order = min(order_cap, len(x0s))
            row["noisy_latents"] = self._predictor(
                sample,
                sigma_t,
                x.new_tensor(sigma_next),
                x0s,
                [x.new_tensor(v) for v in sigmas],
                this_order,
            )
            row["sigma"] = sigma_next
            x0s, sigmas = x0s[-self.order :], sigmas[-self.order :]
            last_sample, prev_order = sample, this_order

    def estimate_calls(self, row: Row) -> int:
        return self.steps * self.prediction.estimate_calls(row)

    def _b_h(self, hh: torch.Tensor) -> torch.Tensor:
        return torch.expm1(hh) if self.solver_type == "bh2" else hh

    def _predictor(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        x0s: list[torch.Tensor],
        sigmas: list[torch.Tensor],
        order: int,
    ) -> torch.Tensor:
        m0 = x0s[-1]
        alpha_t = 1 - sigma_next
        lambda_s0 = _lambda(sigma)
        h = _lambda(sigma_next) - lambda_s0
        hh = -h
        x_t = sigma_next / sigma * sample - alpha_t * torch.expm1(hh) * m0
        if order == 2:  # the reference uses the simplified rhos_p = [0.5]
            rk = (_lambda(sigmas[-2]) - lambda_s0) / h
            x_t = x_t - alpha_t * self._b_h(hh) * (0.5 * ((x0s[-2] - m0) / rk))
        return x_t

    def _corrector(
        self,
        this_x0: torch.Tensor,
        sigma: torch.Tensor,
        x0s: list[torch.Tensor],
        sigmas: list[torch.Tensor],
        last_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        m0 = x0s[-1]
        sigma_s0 = sigmas[-1]
        alpha_t = 1 - sigma
        lambda_s0 = _lambda(sigma_s0)
        h = _lambda(sigma) - lambda_s0
        rks: list[torch.Tensor] = []
        d1s: list[torch.Tensor] = []
        for i in range(1, order):
            rk = (_lambda(sigmas[-(i + 1)]) - lambda_s0) / h
            rks.append(rk)
            d1s.append((x0s[-(i + 1)] - m0) / rk)
        hh = -h
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
        return (
            sigma / sigma_s0 * last_sample
            - alpha_t * h_phi_1 * m0
            - alpha_t * b_h * (corr_res + rho_last * (this_x0 - m0))
        )
