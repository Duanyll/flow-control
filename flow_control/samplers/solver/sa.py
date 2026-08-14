from collections.abc import Callable
from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StepCallback = Callable[[int], None]


@solver_registry.register("sa")
class SASolver(BaseSolver):
    """Second-order SA-Solver used by the public AWM SD3.5 recipe.

    This is the data-prediction, predictor-evaluate-correct (PEC) variant with
    the few-step coefficient correction from the authors' implementation.
    """

    type: Literal["sa"] = "sa"
    eta: float = 0.4
    stochasticity_start: float = 0.3
    stochasticity_end: float = 0.9
    initial_time: float = 1.0 - 1e-3
    adjust_penultimate_time: bool = True

    @staticmethod
    def _alpha(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @classmethod
    def _lambda(cls, t: torch.Tensor) -> torch.Tensor:
        return torch.log(cls._alpha(t)) - torch.log(t)

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        active = (t >= self.stochasticity_start) & (t <= self.stochasticity_end)
        return torch.where(active, t.new_tensor(self.eta), t.new_zeros(()))

    @staticmethod
    def _positive_exponential_integral(
        polynomial_order: Literal[0, 1],
        interval_start: torch.Tensor,
        interval_end: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 + tau**2
        start = scale * interval_start
        end = scale * interval_end
        exp_delta = torch.exp(-(end - start))
        if polynomial_order == 0:
            return torch.exp(end) * (1.0 - exp_delta) / scale
        return torch.exp(end) * ((end - 1.0) - (start - 1.0) * exp_delta) / scale**2

    @classmethod
    def _gradient_coefficients(
        cls,
        lambda_list: list[torch.Tensor],
        interval_start: torch.Tensor,
        interval_end: torch.Tensor,
        tau: torch.Tensor,
    ) -> list[torch.Tensor]:
        integral_0 = cls._positive_exponential_integral(
            0, interval_start, interval_end, tau
        )
        if len(lambda_list) == 1:
            return [integral_0]
        if len(lambda_list) != 2:
            raise ValueError("The AWM SA-Solver port supports orders 1 and 2.")

        integral_1 = cls._positive_exponential_integral(
            1, interval_start, interval_end, tau
        )
        lambda_0, lambda_1 = lambda_list
        denominator = lambda_0 - lambda_1
        return [
            (integral_1 - lambda_1 * integral_0) / denominator,
            (-integral_1 + lambda_0 * integral_0) / denominator,
        ]

    @classmethod
    def _adams_bashforth_update(
        cls,
        x: torch.Tensor,
        tau: torch.Tensor,
        model_history: list[torch.Tensor],
        time_history: list[torch.Tensor],
        noise: torch.Tensor,
        t: torch.Tensor,
        order: Literal[1, 2],
    ) -> torch.Tensor:
        lambda_t = cls._lambda(t)
        lambda_prev = cls._lambda(time_history[-1])
        h = lambda_t - lambda_prev
        lambda_list = [cls._lambda(time_history[-(i + 1)]) for i in range(order)]
        coefficients = cls._gradient_coefficients(
            lambda_list, lambda_prev, lambda_t, tau
        )

        scale = 1.0 + tau**2
        if order == 2:
            correction = (
                torch.exp(scale * lambda_t)
                * (h**2 / 2.0 - (scale * h - 1.0 + torch.exp(-scale * h)) / scale**2)
                / (lambda_prev - cls._lambda(time_history[-2]))
            )
            coefficients[0] = coefficients[0] + correction
            coefficients[1] = coefficients[1] - correction

        sigma_t = t
        gradient = torch.zeros_like(x)
        factor = scale * sigma_t * torch.exp(-(tau**2) * lambda_t)
        for i, coefficient in enumerate(coefficients):
            gradient = gradient + factor * coefficient * model_history[-(i + 1)]

        noise_variance = 1.0 - torch.exp(-2.0 * tau**2 * h)
        noise_part = sigma_t * torch.sqrt(noise_variance.clamp_min(0.0)) * noise
        sigma_prev = time_history[-1]
        return (
            torch.exp(-(tau**2) * h) * (sigma_t / sigma_prev) * x
            + gradient
            + noise_part
        )

    @classmethod
    def _adams_moulton_update(
        cls,
        x: torch.Tensor,
        tau: torch.Tensor,
        model_history: list[torch.Tensor],
        time_history: list[torch.Tensor],
        noise: torch.Tensor,
        t: torch.Tensor,
        order: Literal[1, 2],
    ) -> torch.Tensor:
        lambda_t = cls._lambda(t)
        lambda_prev = cls._lambda(time_history[-1])
        h = lambda_t - lambda_prev
        times = [*time_history, t]
        lambda_list = [cls._lambda(times[-(i + 1)]) for i in range(order)]
        coefficients = cls._gradient_coefficients(
            lambda_list, lambda_prev, lambda_t, tau
        )

        scale = 1.0 + tau**2
        if order == 2:
            correction = torch.exp(scale * lambda_t) * (
                h / 2.0 - (scale * h - 1.0 + torch.exp(-scale * h)) / (scale**2 * h)
            )
            coefficients[0] = coefficients[0] + correction
            coefficients[1] = coefficients[1] - correction

        sigma_t = t
        gradient = torch.zeros_like(x)
        factor = scale * sigma_t * torch.exp(-(tau**2) * lambda_t)
        for i, coefficient in enumerate(coefficients):
            gradient = gradient + factor * coefficient * model_history[-(i + 1)]

        noise_variance = 1.0 - torch.exp(-2.0 * tau**2 * h)
        noise_part = sigma_t * torch.sqrt(noise_variance.clamp_min(0.0)) * noise
        sigma_prev = time_history[-1]
        return (
            torch.exp(-(tau**2) * h) * (sigma_t / sigma_prev) * x
            + gradient
            + noise_part
        )

    @staticmethod
    def _randn_like(
        value: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        return torch.randn(
            value.shape,
            device=value.device,
            dtype=value.dtype,
            generator=generator,
        )

    def sample(
        self,
        model_fn: VelocityFn,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        generator: torch.Generator | None = None,
        step_callback: StepCallback | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if sigmas.ndim != 1 or sigmas.numel() < 3:
            raise ValueError("SA-Solver requires a one-dimensional sigma grid.")

        times = sigmas.clone()
        times[0] = self.initial_time
        if self.adjust_penultimate_time:
            times[-2] = times[-3] / 2.0

        steps = times.numel() - 1
        x = latents
        # Preserve the reference implementation's RNG stream; it draws one
        # unused tensor before the initial model evaluation.
        self._randn_like(x, generator)
        time_history = [times[0]]
        velocity = model_fn(x, times[0])
        model_history = [x - times[0] * velocity]
        intermediates = [x]

        # The public implementation disables the first corrector while building
        # enough history for the second-order method.
        t = times[1]
        noise = self._randn_like(x, generator)
        x = self._adams_bashforth_update(
            x,
            self._tau(t),
            model_history,
            time_history,
            noise,
            t,
            order=1,
        )
        velocity = model_fn(x, t)
        model_history.append(x - t * velocity)
        time_history.append(t)
        intermediates.append(x)
        if step_callback is not None:
            step_callback(1)

        for step in range(2, steps + 1):
            t = times[step]
            if step == steps:
                x = model_history[-1]
            else:
                noise = self._randn_like(x, generator)
                tau = self._tau(t)
                predicted = self._adams_bashforth_update(
                    x,
                    tau,
                    model_history,
                    time_history,
                    noise,
                    t,
                    order=2,
                )
                velocity = model_fn(predicted, t)
                model_history.append(predicted - t * velocity)
                x = self._adams_moulton_update(
                    x,
                    tau,
                    model_history,
                    time_history,
                    noise,
                    t,
                    order=2,
                )
                time_history.append(t)
                del model_history[0]

            intermediates.append(x)
            if step_callback is not None:
                step_callback(step)

        return x, intermediates

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
        raise RuntimeError("SASolver must run through Sampler.sample().")


if __name__ == "__main__":
    from rich import print

    solver = SASolver()
    test_latents = torch.randn(2, 4, 8)
    test_sigmas = torch.linspace(1.0, 0.0, 15)

    def test_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.1 * x + t

    output, trajectory = solver.sample(test_model, test_latents, test_sigmas)
    assert output.shape == test_latents.shape
    assert len(trajectory) == test_sigmas.numel()
    assert torch.isfinite(output).all()
    print("SA-Solver smoke test passed.")
