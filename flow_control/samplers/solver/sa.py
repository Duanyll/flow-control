from dataclasses import dataclass
from typing import Literal

import torch

from ..plan import (
    EvalRequest,
    GuidanceOutput,
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
class SaTransition(Transition):
    """SA transition over the adjusted time grid.

    A grid starting above ``initial_time`` is capped there; the optional
    penultimate adjustment may also change ``sigma_next``. ``eta`` is the
    actual ``tau(sigma_next)`` compiled at plan time (terminal is zero).
    """

    final: bool = False
    """Terminal step: collapse to the previous eval's x0 prediction."""

    def eval_topology(self) -> str:
        # ``final`` collapses the transition to zero evals, so a front slice
        # of a longer plan (no terminal marker) must not fingerprint-match a
        # fresh plan of the same length that ends with one.
        return f"{type(self).__name__}:{type(self.solver).__name__}:final={self.final}"


@dataclass(frozen=True, slots=True)
class SaRuntimeState(SolverRuntimeState):
    """Rolling PEC history (most recent last), rebuilt per transition."""

    model_history: tuple[torch.Tensor, ...]
    """x0 predictions evaluated at the pre-corrector (predicted) points."""
    time_history: tuple[float, ...]


@solver_registry.register("sa")
class SASolver(BaseSolver):
    """Second-order SA-Solver used by the public AWM SD3.5 recipe.

    This is the data-prediction, predictor-evaluate-correct (PEC) variant with
    the few-step coefficient correction from the authors' implementation. Each
    PEC transition evaluates the model once at the predicted point (plus one
    seeding eval at the current point when the history is empty); the random
    PEC step has no simple step-wise Gaussian density, so it does not support
    log-prob replay.
    """

    type: Literal["sa"] = "sa"
    eta: float = 0.4
    stochasticity_start: float = 0.3
    stochasticity_end: float = 0.9
    initial_time: float = 1.0 - 1e-3
    adjust_penultimate_time: bool = True

    def _tau(self, t: float) -> float:
        if self.stochasticity_start <= t <= self.stochasticity_end:
            return self.eta
        return 0.0

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        if len(sigmas) < 3:
            raise ValueError("SA-Solver requires a sigma grid of at least 3 entries.")
        times = list(sigmas)
        # The reference avoids evaluating exactly at pure noise for a full
        # 1.0-start grid. A partial plan already below that point must retain
        # the exact sigma supplied by its init operation.
        times[0] = min(times[0], self.initial_time)
        if self.adjust_penultimate_time:
            times[-2] = times[-3] / 2.0
        num_steps = len(times) - 1
        return [
            SaTransition(
                solver=self,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=self._tau(sigma_next) if index < num_steps - 1 else 0.0,
                final=index == num_steps - 1,
            )
            for index, (sigma, sigma_next) in enumerate(
                zip(times[:-1], times[1:], strict=True)
            )
        ]

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        assert isinstance(tr, SaTransition)
        state = ctx.solver_state
        assert state is None or isinstance(state, SaRuntimeState)
        latents = ctx.latents

        if state is None:
            # Empty history (run start or sliced plan): seed it with an eval
            # at the current point, mirroring the reference initial evaluation.
            out = yield EvalRequest(latents=latents, sigma=tr.sigma)
            x0 = self._velocity_to_x0(
                out.velocity, latents, latents.new_tensor(tr.sigma)
            )
            model_history: tuple[torch.Tensor, ...] = (x0,)
            time_history: tuple[float, ...] = (tr.sigma,)
        else:
            model_history = state.model_history
            time_history = state.time_history

        if tr.final:
            # Reference final step: return the last x0 prediction directly,
            # without predictor, corrector or noise.
            next_latents = model_history[-1]
            recorded = (
                self._make_recorded_step(tr, ctx, next_latents, zero_log_prob(latents))
                if tr.record
                else None
            )
            return TransitionResult(next_latents=next_latents, recorded=recorded)

        t = latents.new_tensor(tr.sigma_next)
        tau = latents.new_tensor(tr.eta)
        # The reference disables the corrector on the first (order-1) step
        # while the second-order history builds up.
        order: Literal[1, 2] = 1 if len(model_history) < 2 else 2
        noise = (
            torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=ctx.generator,
            )
            if tr.eta > 0.0
            else torch.zeros_like(latents)
        )
        model_list = list(model_history)
        time_list = [latents.new_tensor(value) for value in time_history]

        predicted = self._adams_bashforth_update(
            latents, tau, model_list, time_list, noise, t, order=order
        )
        out = yield EvalRequest(latents=predicted, sigma=tr.sigma_next)
        new_model = self._velocity_to_x0(out.velocity, predicted, t)
        if order == 1:
            next_latents = predicted
        else:
            next_latents = self._adams_moulton_update(
                latents, tau, [*model_list, new_model], time_list, noise, t, order=2
            )

        next_state = SaRuntimeState(
            model_history=(*model_history, new_model)[-2:],
            time_history=(*time_history, tr.sigma_next)[-2:],
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
    def _alpha(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @classmethod
    def _lambda(cls, t: torch.Tensor) -> torch.Tensor:
        return torch.log(cls._alpha(t)) - torch.log(t)

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


if __name__ == "__main__":
    from rich import print

    solver = SASolver()
    plan = solver.plan(torch.linspace(1.0, 0.0, 15).tolist())
    context = StepContext(
        latents=torch.randn(2, 4, 8),
        generator=torch.Generator().manual_seed(0),
        solver_state=None,
        guidance_state=None,
    )
    eval_count = 0
    for item in plan:
        generator_obj = item.run(context)
        try:
            request = next(generator_obj)
            while True:
                eval_count += 1
                velocity = 0.1 * request.latents + request.sigma
                request = generator_obj.send(GuidanceOutput(velocity=velocity))
        except StopIteration as stop:
            result = stop.value
        context.latents = result.next_latents
        if result.next_solver_state is not None:
            context.solver_state = result.next_solver_state
    assert eval_count == len(plan)  # 2 seeding+predicted, then 1 each, final 0
    assert torch.isfinite(context.latents).all()
    print("SA-Solver plan execution smoke test passed.")
