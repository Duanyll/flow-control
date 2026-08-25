"""Plan/transform/executor semantics for the plan-as-data sampler.

Two kinds of test live here:

* **Parity** (``BaselineParityTest``) replays the numeric fixtures under
  ``tests/fixtures/sampler_baselines/`` — first captured from the pre
  plan-as-data implementation — so accidental numeric drift in any solver
  surfaces as a bitwise mismatch. Regenerate with
  ``tests/sampler_baseline_capture.py`` only when behaviour changes on purpose.
* **Invariants** assert properties of the plan data or compare two executions
  of the current code; they need no fixtures.
"""

import importlib
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from flow_control.samplers import (
    PhaseConfig,
    PhasesRecipe,
    RecipeBuildContext,
    SdeWindow,
    run_phases,
)
from flow_control.samplers.executor import Run, execute
from flow_control.samplers.plan import (
    EvalRequest,
    GuidanceOutput,
    RecordedStep,
    StepContext,
    Transition,
    TransitionResult,
)
from flow_control.samplers.sampler import (
    ReplayItem,
    SampleOutput,
    Sampler,
    SampleRequest,
)
from flow_control.samplers.solver import (
    CPSSolver,
    DanceSolver,
    DDIMSolver,
    DPMSolver,
    FlashSolver,
    FlowSolver,
    FlowUniPCSolver,
    SASolver,
    solver_registry,
)
from flow_control.samplers.solver.flash import FlashReplayStep, FlashTransition
from flow_control.samplers.solver.sa import SaRuntimeState, SaTransition
from flow_control.samplers.transforms import (
    finalize_replay_state,
    select_sde_window,
    with_sde_window,
)
from flow_control.utils.tensor import deep_apply_tensor_fn, deep_move_to_device

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
capture = importlib.import_module("sampler_baseline_capture")
microbatching = importlib.import_module("test_microbatching")

BASELINE_DIR = capture.BASELINE_DIR
RecordingModel = capture.RecordingModel
make_request_batch = capture.make_request_batch
drive_single_eval_transition = capture.drive_single_eval_transition

MIGRATED_STEP_FIXTURES = {
    # Each solver exposes its step moments as ``step_parts(latents, velocity,
    # sigma, sigma_next, eta) -> (mean, std_dev, ...)``.
    "step_flow_eta07.pt": FlowSolver,
    "step_ddim_eta07.pt": DDIMSolver,
    "step_cps_eta07.pt": CPSSolver,
    "step_dance_eta07.pt": DanceSolver,
}


def load_fixture(name: str) -> dict:
    return torch.load(BASELINE_DIR / name, weights_only=True)


def assert_bitwise(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def recording_recipe(**window_kwargs) -> PhasesRecipe:
    """A single-phase recipe recording the (default: full) SDE window."""
    return PhasesRecipe(
        phases=[PhaseConfig(transforms=[SdeWindow(record=True, **window_kwargs)])]
    )


def run_recipe(
    model,
    sampler: Sampler,
    batch,
    generator: torch.Generator | None,
    recipe: PhasesRecipe | None = None,
) -> SampleOutput:
    """Recipe-runner replacement for the deleted sample(return_trajectory=True)."""
    phases = (recipe or recording_recipe()).build(
        RecipeBuildContext(
            default_sampler=sampler,
            batches={"main": batch},
            negative_batch_for=lambda name: None,
            generator=generator,
        )
    )
    return run_phases(model, [phases])[0]


class ConstVelocityModel:
    """Duck-typed model returning a fixed velocity for every sample."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, velocity: torch.Tensor) -> None:
        self.velocity = velocity

    def predict_velocity_batched(
        self,
        batches: list,
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [self.velocity for _ in batches]


def stack_trajectory(output: SampleOutput) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the stacked ``(latents, log_probs)`` fixture views."""
    trajectory = output.trajectory
    assert trajectory
    latents = torch.stack(
        [trajectory[0].latent_t] + [step.latent_next for step in trajectory],
        dim=1,
    )
    log_probs = torch.stack([step.log_prob for step in trajectory], dim=1)
    return latents, log_probs


def sampler_from_fixture(fixture: dict) -> Sampler:
    """Rebuild the Sampler from a frozen config dump.

    Fixtures captured before the guidance/recipe rename store the retired
    spellings; they are migrated here exactly as a user would have to edit an
    old config. Regenerating a fixture writes the current spelling and makes
    the corresponding branch a no-op.
    """
    config = dict(fixture["sampler_config"])
    if "cfg_scale" in config:
        config["guidance"] = {
            "type": "cfg",
            "scale": config.pop("cfg_scale"),
            "renorm": config.pop("enable_cfg_renorm"),
            "renorm_eps": config.pop("cfg_renorm_eps"),
            "renorm_min": config.pop("cfg_renorm_min"),
        }
    for key in ("trajectory_window_size", "trajectory_window_range"):
        if key in config:
            assert config.pop(key) is None, f"unexpected frozen window in {key}"
    shift = config.get("shift")
    if isinstance(shift, dict) and shift.get("type") == "none":
        # The `none` shift was deleted as an exact duplicate of ConstantShift's
        # default factor of 1.0, so the baseline numbers are unaffected.
        config["shift"] = {**shift, "type": "constant"}
    return Sampler.model_validate(config)


def run_e2e_fixture(fixture: dict) -> tuple[SampleOutput, Any]:
    """Re-run a captured end-to-end config through the current recipe runner.

    The second element is the ``RecordingModel`` that answered the evals; it is
    untyped because the capture module is imported dynamically (see above).
    """
    model = RecordingModel()
    output = run_recipe(
        model,
        sampler_from_fixture(fixture),
        make_request_batch(fixture["initial_latents"]),
        torch.Generator().manual_seed(777),
    )
    return output, model


def e2e_fixtures() -> list[Path]:
    return sorted(BASELINE_DIR.glob("e2e_*.pt"))


def drive_transition(
    tr: Transition,
    ctx: StepContext,
    velocity_fn,
) -> tuple[TransitionResult, list[EvalRequest]]:
    """Drive a run_transition generator, answering evals via ``velocity_fn``."""
    gen = tr.run(ctx)
    requests: list[EvalRequest] = []
    try:
        request = next(gen)
        while True:
            requests.append(request)
            request = gen.send(GuidanceOutput(velocity=velocity_fn(request)))
    except StopIteration as stop:
        assert isinstance(stop.value, TransitionResult)
        return stop.value, requests


class BaselineParityTest(unittest.TestCase):
    """Bitwise parity against the checked-in numeric baselines.

    A failure here means a solver's numerics moved. If that was intentional,
    regenerate the fixtures (see ``tests/sampler_baseline_capture.py``) and
    commit them with the change; if not, it is the regression these exist for.
    """

    def test_e2e_trajectories_match_baselines(self) -> None:
        for fixture_path in e2e_fixtures():
            with self.subTest(fixture=fixture_path.stem):
                fixture = load_fixture(fixture_path.name)
                output, _ = run_e2e_fixture(fixture)

                assert_bitwise(output.final_latents, fixture["final_latents"])
                traj_latents, log_probs = stack_trajectory(output)
                if fixture["log_probs"] is None:
                    # Captured by the legacy SA loop, which stored every
                    # intermediate latent, produced no log-probs, and reported
                    # the raw config grid rather than SA's adjusted one (its
                    # head/penultimate rewrites are pinned by the sa_internals
                    # tests). Regenerating this fixture retires the branch.
                    assert_bitwise(output.timesteps[1:-1], fixture["timesteps"][1:-1])
                    window = traj_latents.shape[1]
                    assert_bitwise(traj_latents, fixture["traj_latents"][:, :window])
                    assert_bitwise(output.final_latents, fixture["traj_latents"][:, -1])
                    self.assertTrue((log_probs == 0).all())
                else:
                    assert_bitwise(output.timesteps, fixture["timesteps"])
                    assert_bitwise(traj_latents, fixture["traj_latents"])
                    assert_bitwise(log_probs, fixture["log_probs"])

    def test_e2e_eval_topology_matches_baselines(self) -> None:
        # Separate from the numerics: *where* and in what order the model is
        # queried is a property of the plan, and can change (extra warmup eval,
        # reordered rendezvous) while the final latents still land correctly.
        for fixture_path in e2e_fixtures():
            with self.subTest(fixture=fixture_path.stem):
                fixture = load_fixture(fixture_path.name)
                _, model = run_e2e_fixture(fixture)

                self.assertEqual(model.eval_sigmas, fixture["eval_sigmas"])
                for actual, expected in zip(
                    model.eval_latents, fixture["eval_latents"], strict=True
                ):
                    assert_bitwise(actual, expected)

    def test_migrated_transitions_match_stochastic_step_baselines(self) -> None:
        for fixture_name, solver_cls in MIGRATED_STEP_FIXTURES.items():
            fixture = load_fixture(fixture_name)
            solver = solver_cls.model_validate(fixture["solver_config"])
            latents = fixture["latents"]
            velocity = fixture["velocity"]
            for entry in fixture["entries"]:
                with self.subTest(fixture=fixture_name, sigma=entry["sigma"]):
                    tr = Transition(
                        solver=solver,
                        sigma=entry["sigma"],
                        sigma_next=entry["sigma_next"],
                        eta=solver.eta,
                        record=True,
                    )
                    ctx = StepContext(
                        latents=latents,
                        generator=torch.Generator().manual_seed(
                            5000 + entry["step_index"]
                        ),
                        solver_state=None,
                        guidance_state=None,
                    )
                    result = drive_single_eval_transition(tr, ctx, velocity)
                    assert_bitwise(result.next_latents, entry["next_latents"])
                    assert result.recorded is not None
                    assert_bitwise(result.recorded.log_prob, entry["log_prob"])

                    parts = solver_cls.step_parts(
                        latents, velocity, tr.sigma, tr.sigma_next, tr.eta
                    )
                    assert_bitwise(parts[0], entry["mean"])
                    assert_bitwise(parts[1], entry["std_dev"])

    def test_replay_steps_match_replay_baselines(self) -> None:
        for fixture_name, solver_cls in MIGRATED_STEP_FIXTURES.items():
            fixture = load_fixture(fixture_name)
            solver = solver_cls.model_validate(fixture["solver_config"])
            for entry in fixture["entries"]:
                with self.subTest(fixture=fixture_name, sigma=entry["sigma"]):
                    replay = solver.make_replay(
                        entry["sigma"], entry["sigma_next"], solver.eta
                    )
                    output = replay.logprob(
                        fixture["velocity"],
                        fixture["latents"],
                        fixture["prev_sample"],
                    )
                    assert_bitwise(output.log_prob, entry["replay_log_prob"])
                    assert_bitwise(output.mean, entry["replay_mean"])
                    assert_bitwise(output.std_dev, entry["replay_std_dev"])

    def test_replay_recorded_steps_matches_replay_baselines(self) -> None:
        for fixture_name, solver_cls in MIGRATED_STEP_FIXTURES.items():
            fixture = load_fixture(fixture_name)
            solver_type = solver_registry.get(fixture["solver_config"]["type"])
            self.assertIs(solver_type, solver_cls)
            solver = solver_cls.model_validate(fixture["solver_config"])
            sampler = Sampler(solver=solver)
            model = ConstVelocityModel(fixture["velocity"])
            items = [
                ReplayItem(
                    batch=make_request_batch(fixture["latents"]),
                    recorded=RecordedStep(
                        latent_t=fixture["latents"],
                        latent_next=fixture["prev_sample"],
                        log_prob=torch.zeros(1),
                        replay=solver.make_replay(
                            entry["sigma"], entry["sigma_next"], solver.eta
                        ),
                    ),
                )
                for entry in fixture["entries"]
            ]
            outputs = sampler.replay_recorded_steps(model, items)  # type: ignore[arg-type]
            for entry, output in zip(fixture["entries"], outputs, strict=True):
                with self.subTest(fixture=fixture_name, sigma=entry["sigma"]):
                    assert_bitwise(output.log_prob, entry["replay_log_prob"])
                    assert_bitwise(output.mean, entry["replay_mean"])
                    assert_bitwise(output.std_dev, entry["replay_std_dev"])

    def test_flash_transitions_match_step_baselines(self) -> None:
        fixture = load_fixture("step_flash_eta1.pt")
        solver = FlashSolver.model_validate(fixture["solver_config"])
        latents = fixture["latents"]
        velocity = fixture["velocity"]
        # The fixture sweep was captured on the full 8-step grid; the compiled
        # per-step noise scale must match the legacy step-index lerp bitwise.
        plan = [
            item
            for item in solver.plan(torch.linspace(1.0, 0.0, 9).tolist())
            if isinstance(item, FlashTransition)
        ]
        self.assertEqual(len(plan), len(fixture["entries"]))
        self.assertEqual(plan[-1].eta, 0.0)

        for tr, entry in zip(plan, fixture["entries"], strict=True):
            with self.subTest(sigma=entry["sigma"]):
                index = entry["step_index"]
                self.assertEqual(tr.sigma, entry["sigma"])
                self.assertEqual(tr.sigma_next, entry["sigma_next"])
                deterministic = tr.eta == 0.0 or tr.sigma_next <= 0.0
                if not deterministic:
                    # std_dev pins the compiled noise scale against the sweep.
                    assert_bitwise(
                        latents.new_tensor(tr.sigma_next) * tr.noise_scale,
                        entry["std_dev"],
                    )
                    assert_bitwise(
                        FlashSolver.renoise_parts(
                            latents, velocity, tr.sigma, tr.sigma_next, tr.noise_scale
                        )[0],
                        entry["mean"],
                    )
                # The fixture captured every entry with eta=1; the terminal
                # entry is deterministic via sigma_next == 0 in both paths.
                ctx = StepContext(
                    latents=latents,
                    generator=torch.Generator().manual_seed(5000 + index),
                    solver_state=None,
                    guidance_state=None,
                )
                result = drive_single_eval_transition(
                    replace(tr, record=True), ctx, velocity
                )
                assert_bitwise(result.next_latents, entry["next_latents"])
                assert result.recorded is not None
                assert_bitwise(result.recorded.log_prob, entry["log_prob"])

                replay = result.recorded.replay
                assert isinstance(replay, FlashReplayStep)
                self.assertEqual(replay.noise_scale, tr.noise_scale)
                output = replay.logprob(velocity, latents, fixture["prev_sample"])
                assert_bitwise(output.log_prob, entry["replay_log_prob"])
                assert_bitwise(output.mean, entry["replay_mean"])
                assert_bitwise(output.std_dev, entry["replay_std_dev"])

    def test_flash_public_replay_matches_step_baselines(self) -> None:
        # The recorded FlashReplayStep carries the plan-compiled noise scale;
        # no runtime state is needed at replay time.
        fixture = load_fixture("step_flash_eta1.pt")
        solver = FlashSolver.model_validate(fixture["solver_config"])
        sampler = Sampler(steps=8, solver=solver)
        model = ConstVelocityModel(fixture["velocity"])
        items = [
            ReplayItem(
                batch=make_request_batch(fixture["latents"]),
                recorded=RecordedStep(
                    latent_t=fixture["latents"],
                    latent_next=fixture["prev_sample"],
                    log_prob=torch.zeros(1),
                    replay=FlashReplayStep(
                        sigma=entry["sigma"],
                        sigma_next=entry["sigma_next"],
                        eta=solver.eta,
                        noise_scale=solver._noise_scale_at(entry["step_index"], 8),
                    ),
                ),
            )
            for entry in fixture["entries"]
        ]
        outputs = sampler.replay_recorded_steps(model, items)  # type: ignore[arg-type]
        for entry, output in zip(fixture["entries"], outputs, strict=True):
            with self.subTest(sigma=entry["sigma"]):
                assert_bitwise(output.log_prob, entry["replay_log_prob"])
                assert_bitwise(output.mean, entry["replay_mean"])
                assert_bitwise(output.std_dev, entry["replay_std_dev"])

    def test_sa_plan_compiles_adjusted_grid_and_taus(self) -> None:
        # The self-contained companion in PlanInvariantsTest derives the same
        # grid from SA's documented rules; this one pins it against the numbers
        # the pre-refactor planner actually produced.
        fixture = load_fixture("sa_internals.pt")
        solver = SASolver.model_validate(fixture["solver_config"])
        plan = [
            item
            for item in solver.plan(fixture["sigmas"].tolist())
            if isinstance(item, SaTransition)
        ]
        times = fixture["times"]
        self.assertEqual(len(plan), times.numel() - 1)
        # Transitions run on the adjusted grid (initial_time head, penultimate
        # halving); compare after float32 rounding, as executed.
        assert_bitwise(
            torch.tensor([item.sigma for item in plan], dtype=torch.float32),
            times[:-1],
        )
        assert_bitwise(
            torch.tensor([item.sigma_next for item in plan], dtype=torch.float32),
            times[1:],
        )
        # Per-step eta is the actual tau at the target time; terminal is zero.
        assert_bitwise(
            torch.tensor([item.eta for item in plan], dtype=torch.float32)[:-1],
            fixture["tau_values"][1:-1],
        )
        self.assertEqual(plan[-1].eta, 0.0)
        self.assertTrue(plan[-1].final)
        self.assertFalse(any(item.final for item in plan[:-1]))

    def test_sa_internal_updates_still_match_baselines(self) -> None:
        # The shared AB/AM math is unchanged; pin it against the old fixtures.
        fixture = load_fixture("sa_internals.pt")
        solver = SASolver.model_validate(fixture["solver_config"])
        times = fixture["times"]
        x = fixture["x"]
        tau_values = torch.stack([x.new_tensor(solver._tau(float(t))) for t in times])
        assert_bitwise(tau_values, fixture["tau_values"])

        ab1 = solver._adams_bashforth_update(
            x,
            x.new_tensor(solver._tau(float(times[1]))),
            [fixture["m0"]],
            [times[0]],
            fixture["noise"],
            times[1],
            order=1,
        )
        assert_bitwise(ab1, fixture["ab1"])

        for entry in fixture["entries"]:
            with self.subTest(t_index=entry["t_index"]):
                t = times[entry["t_index"]]
                tau = x.new_tensor(solver._tau(float(t)))
                assert_bitwise(tau, entry["tau"])
                time_history = [
                    times[entry["t_index"] - 2],
                    times[entry["t_index"] - 1],
                ]
                ab2 = solver._adams_bashforth_update(
                    x,
                    tau,
                    [fixture["m0"], fixture["m1"]],
                    time_history,
                    fixture["noise"],
                    t,
                    order=2,
                )
                assert_bitwise(ab2, entry["ab2"])
                am2 = solver._adams_moulton_update(
                    x,
                    tau,
                    [fixture["m0"], fixture["m1"], fixture["m2"]],
                    time_history,
                    fixture["noise"],
                    t,
                    order=2,
                )
                assert_bitwise(am2, entry["am2"])

    def test_sa_run_transition_drives_pec_through_internal_updates(self) -> None:
        # Drive run_transition generators with the fixture histories and check
        # the PEC control flow reproduces the pinned AB/AM outputs.
        fixture = load_fixture("sa_internals.pt")
        solver = SASolver.model_validate(fixture["solver_config"])
        times = fixture["times"]
        x = fixture["x"]
        m0, m1 = fixture["m0"], fixture["m1"]

        # First executed transition with a seeded one-entry history: order-1
        # AB predictor, eval at the predicted point, no corrector.
        tr = SaTransition(
            solver=solver,
            sigma=float(times[0]),
            sigma_next=float(times[1]),
            eta=solver._tau(float(times[1])),
        )
        sent_velocity = torch.full_like(x, 0.25)
        ctx = StepContext(
            latents=x,
            generator=torch.Generator().manual_seed(32),
            solver_state=SaRuntimeState(
                model_history=(m0,), time_history=(float(times[0]),)
            ),
            guidance_state=None,
        )
        result, requests = drive_transition(tr, ctx, lambda request: sent_velocity)
        self.assertEqual(len(requests), 1)
        assert_bitwise(requests[0].latents, fixture["ab1"])
        self.assertEqual(requests[0].sigma, tr.sigma_next)
        assert_bitwise(result.next_latents, fixture["ab1"])
        state = result.next_solver_state
        assert isinstance(state, SaRuntimeState)
        assert_bitwise(state.model_history[0], m0)
        assert_bitwise(
            state.model_history[1],
            fixture["ab1"] - x.new_tensor(tr.sigma_next) * sent_velocity,
        )
        self.assertEqual(state.time_history, (float(times[0]), float(times[1])))

        # Order-2 PEC at an in-window time (tau = eta) and outside it (tau = 0).
        for entry in fixture["entries"]:
            with self.subTest(t_index=entry["t_index"]):
                t_index = entry["t_index"]
                tr = SaTransition(
                    solver=solver,
                    sigma=float(times[t_index - 1]),
                    sigma_next=float(times[t_index]),
                    eta=solver._tau(float(times[t_index])),
                )
                generator = torch.Generator().manual_seed(32)
                ctx = StepContext(
                    latents=x,
                    generator=generator,
                    solver_state=SaRuntimeState(
                        model_history=(m0, m1),
                        time_history=(
                            float(times[t_index - 2]),
                            float(times[t_index - 1]),
                        ),
                    ),
                    guidance_state=None,
                )
                result, requests = drive_transition(
                    tr, ctx, lambda request: sent_velocity
                )
                self.assertEqual(len(requests), 1)
                assert_bitwise(requests[0].latents, entry["ab2"])
                if tr.eta == 0.0:
                    # Deterministic PEC consumes no generator draws.
                    self.assertTrue(
                        torch.equal(
                            generator.get_state(),
                            torch.Generator().manual_seed(32).get_state(),
                        )
                    )
                # The corrector reuses the predictor's noise and the fresh x0
                # evaluated at the predicted point.
                new_model = entry["ab2"] - x.new_tensor(tr.sigma_next) * sent_velocity
                noise = (
                    torch.randn(
                        x.shape,
                        dtype=x.dtype,
                        generator=torch.Generator().manual_seed(32),
                    )
                    if tr.eta > 0.0
                    else torch.zeros_like(x)
                )
                expected = solver._adams_moulton_update(
                    x,
                    x.new_tensor(tr.eta),
                    [m0, m1, new_model],
                    [times[t_index - 2], times[t_index - 1]],
                    noise,
                    times[t_index],
                    order=2,
                )
                assert_bitwise(result.next_latents, expected)


class PlanInvariantsTest(unittest.TestCase):
    def test_with_sde_window_gates_eta_and_marks_record(self) -> None:
        plan = FlowSolver(eta=0.7).plan([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        windowed = [
            item
            for item in with_sde_window(plan, 1, 3, record=True)
            if isinstance(item, Transition)
        ]
        self.assertEqual([item.eta for item in windowed], [0.0, 0.7, 0.7, 0.0, 0.0])
        self.assertEqual(
            [item.record for item in windowed],
            [False, True, True, False, False],
        )
        unrecorded = with_sde_window(plan, 1, 3, record=False)
        self.assertFalse(
            any(item.record for item in unrecorded if isinstance(item, Transition))
        )

    def test_sa_plan_compiles_the_adjusted_grid_and_frozen_tau(self) -> None:
        # SA does not run on the grid it is handed: the head is capped at
        # initial_time (the reference never evaluates at pure noise) and the
        # penultimate time is halved. Both rewrites are invisible on a uniform
        # grid, hence the uneven sigmas here. ``eta`` is tau(sigma_next) frozen
        # at plan time -- zero below the stochasticity band, and zero on the
        # terminal step, which collapses to the previous x0 prediction.
        solver = SASolver()  # eta 0.4 inside [0.3, 0.9]
        transitions = [
            item
            for item in solver.plan([1.0, 0.9, 0.5, 0.2, 0.0])
            if isinstance(item, SaTransition)
        ]

        self.assertEqual(
            [item.sigma for item in transitions],
            [solver.initial_time, 0.9, 0.5, 0.25],
        )
        self.assertEqual(
            [item.sigma_next for item in transitions], [0.9, 0.5, 0.25, 0.0]
        )
        self.assertEqual([item.eta for item in transitions], [0.4, 0.4, 0.0, 0.0])
        self.assertEqual(
            [item.final for item in transitions], [False, False, False, True]
        )


class ExecutorSemanticsTest(unittest.TestCase):
    def test_ddim_eta0_consumes_no_generator_draws(self) -> None:
        sampler = Sampler(steps=3, solver=DDIMSolver())
        generator = torch.Generator().manual_seed(5)
        state_before = generator.get_state().clone()
        sampler.sample(
            microbatching.FakeSamplerModel(),
            [
                SampleRequest(
                    batch=microbatching.make_sampler_batch(0.5),
                    generator=generator,
                )
            ],
        )
        self.assertTrue(torch.equal(generator.get_state(), state_before))

    def test_recording_changes_neither_the_path_nor_the_reported_grid(self) -> None:
        # sde_window gates eta whether or not it records: recording must not
        # change the sampled path (same RNG consumption), and the executed
        # sigma grid is plan metadata (full sigmas[:-1], never window-sliced).
        def run(record: bool) -> SampleOutput:
            return run_recipe(
                microbatching.FakeSamplerModel(),
                Sampler(steps=6, solver=FlowSolver(eta=0.7)),
                microbatching.make_sampler_batch(0.4),
                torch.Generator().manual_seed(11),
                recipe=PhasesRecipe(
                    phases=[
                        PhaseConfig(
                            transforms=[SdeWindow(size=2, range=(1, 5), record=record)]
                        )
                    ]
                ),
            )

        recorded = run(True)
        plain = run(False)
        assert_bitwise(recorded.final_latents, plain.final_latents)
        self.assertEqual(recorded.timesteps.numel(), 6)
        assert_bitwise(recorded.timesteps, plain.timesteps)
        self.assertIsNone(plain.trajectory)
        # Recorded steps chain: each latent_next is the next step's latent_t.
        trajectory = recorded.trajectory
        assert trajectory is not None
        self.assertEqual(len(trajectory), 2)
        assert_bitwise(trajectory[0].latent_next, trajectory[1].latent_t)

    def test_rollout_and_replay_are_self_consistent(self) -> None:
        # GRPO-shaped parity: replay every RecordedStep through the public
        # entry point; the deterministic fake model reproduces the rollout
        # velocity, so the replayed log-prob must match bitwise.
        solvers = [
            FlowSolver(eta=0.5),
            DDIMSolver(eta=0.5),
            CPSSolver(eta=0.5),
            DanceSolver(eta=0.5),
            FlashSolver(noise_scale_start=0.9, noise_scale_end=0.7),
        ]
        velocity = torch.full((1, 4, 2), 0.3)
        for solver in solvers:
            with self.subTest(solver=solver.type):
                sampler = Sampler(steps=4, solver=solver)
                batch = microbatching.make_batch(0.9)
                model = ConstVelocityModel(velocity)
                output = run_recipe(
                    model, sampler, batch, torch.Generator().manual_seed(3)
                )
                trajectory = output.trajectory
                assert trajectory is not None
                self.assertEqual(len(trajectory), sampler.steps - 1)
                self.assertTrue(all((step.log_prob != 0).all() for step in trajectory))

                items = [ReplayItem(batch=batch, recorded=step) for step in trajectory]
                replayed = sampler.replay_recorded_steps(model, items)  # type: ignore[arg-type]
                for step, replay in zip(trajectory, replayed, strict=True):
                    assert_bitwise(replay.log_prob, step.log_prob)

                # Replay reads the ReplayStep's stored ACTUAL eta, never the
                # sampler's live solver config.
                retuned = Sampler(steps=4, solver=type(solver)(eta=0.123))
                for output_now, output_then in zip(
                    retuned.replay_recorded_steps(model, items),  # type: ignore[arg-type]
                    replayed,
                    strict=True,
                ):
                    assert_bitwise(output_now.log_prob, output_then.log_prob)

    def test_recorded_steps_round_trip_deep_move_to_device(self) -> None:
        velocity = torch.full((1, 4, 2), 0.3)
        sampler = Sampler(steps=4, solver=FlowSolver(eta=0.5))
        batch = microbatching.make_batch(0.9)
        model = ConstVelocityModel(velocity)
        output = run_recipe(model, sampler, batch, torch.Generator().manual_seed(3))
        assert output.trajectory is not None

        # CPU rollout storage round-trips the whole SampleOutput dataclass,
        # rebuilding fresh RecordedStep/ReplayStep instances along the way.
        moved: SampleOutput = deep_move_to_device(
            output, torch.device("cpu"), preserve_aliases=True
        )
        self.assertIsNot(moved, output)
        assert moved.trajectory is not None
        for original, restored in zip(output.trajectory, moved.trajectory, strict=True):
            self.assertIsNot(restored, original)
            self.assertEqual(restored.replay, original.replay)
            assert_bitwise(restored.latent_t, original.latent_t)
            assert_bitwise(restored.latent_next, original.latent_next)
        for previous, following in zip(
            moved.trajectory[:-1], moved.trajectory[1:], strict=True
        ):
            self.assertIs(previous.latent_next, following.latent_t)

        items = [ReplayItem(batch=batch, recorded=step) for step in moved.trajectory]
        replayed = sampler.replay_recorded_steps(model, items)  # type: ignore[arg-type]
        for step, replay in zip(output.trajectory, replayed, strict=True):
            assert_bitwise(replay.log_prob, step.log_prob)

    def test_deep_tensor_map_alias_table_is_per_call(self) -> None:
        tensor = torch.ones(1)
        transforms: list[torch.Tensor] = []

        def clone(value: torch.Tensor) -> torch.Tensor:
            transformed = value.clone()
            transforms.append(transformed)
            return transformed

        first = deep_apply_tensor_fn([tensor, tensor], clone, preserve_aliases=True)
        second = deep_apply_tensor_fn([tensor, tensor], clone, preserve_aliases=True)

        self.assertEqual(len(transforms), 2)
        self.assertIs(first[0], first[1])
        self.assertIs(second[0], second[1])
        self.assertIsNot(first[0], second[0])

    def test_flash_windowed_recording_stores_offset_noise_scales(self) -> None:
        # With an sde_window the recorded steps start at train_start > 0; each
        # FlashReplayStep must carry the plan-compiled noise scale of its
        # plan-level step index (which varies per step here, so an offset bug
        # would change the numbers).
        velocity = torch.full((1, 4, 2), 0.3)
        solver = FlashSolver(noise_scale_start=0.9, noise_scale_end=0.7)
        sampler = Sampler(steps=8, solver=solver)
        batch = microbatching.make_batch(0.9)
        model = ConstVelocityModel(velocity)
        output = run_recipe(
            model,
            sampler,
            batch,
            torch.Generator().manual_seed(3),
            recipe=recording_recipe(size=3, range=(2, 7)),
        )
        trajectory = output.trajectory
        assert trajectory is not None

        # The window is drawn from the request generator before any sampling
        # randomness, so a fresh generator with the same seed reproduces it.
        train_start, train_end = select_sde_window(
            sampler.steps, 3, (2, 7), torch.Generator().manual_seed(3)
        )
        self.assertGreater(train_start, 0)
        self.assertEqual(len(trajectory), train_end - train_start)
        for index, step in enumerate(trajectory):
            replay = step.replay
            assert isinstance(replay, FlashReplayStep)
            self.assertEqual(
                replay.noise_scale,
                solver._noise_scale_at(train_start + index, sampler.steps),
            )

        items = [ReplayItem(batch=batch, recorded=step) for step in trajectory]
        replayed = sampler.replay_recorded_steps(model, items)  # type: ignore[arg-type]
        for step, replay_output in zip(trajectory, replayed, strict=True):
            assert_bitwise(replay_output.log_prob, step.log_prob)

        # A wrong noise scale must change the replayed log-prob.
        first = trajectory[0]
        first_replay = first.replay
        assert isinstance(first_replay, FlashReplayStep)
        corrupted = ReplayItem(
            batch=batch,
            recorded=replace(
                first,
                replay=replace(
                    first_replay,
                    noise_scale=solver._noise_scale_at(0, sampler.steps),
                ),
            ),
        )
        wrong = sampler.replay_recorded_steps(model, [corrupted])[0]  # type: ignore[arg-type]
        self.assertFalse(torch.equal(wrong.log_prob, first.log_prob))

    def test_sliced_plans_warm_up_from_empty_history(self) -> None:
        # New capability: the tail half of a multistep plan runs standalone,
        # warming up from an empty history instead of a compiled step index.
        # The sliced tail must behave exactly like a fresh plan built on the
        # tail sigma grid (same per-step order metadata, same warmup).
        initial = torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(4))
        sigmas = torch.linspace(1.0, 0.0, 9).tolist()
        for solver in (DPMSolver(order=2), FlowUniPCSolver()):
            with self.subTest(solver=solver.type):
                sampler = Sampler(steps=8, solver=solver)
                full = solver.plan(sigmas)
                tail_sliced = finalize_replay_state(full[len(full) // 2 :])
                tail_fresh = finalize_replay_state(
                    solver.plan(sigmas[len(full) // 2 :])
                )
                self.assertEqual(tail_sliced, tail_fresh)

                results: list[torch.Tensor] = []
                for plan in (tail_sliced, tail_fresh):
                    run = Run(
                        plan=plan,
                        ctx=StepContext(
                            latents=initial.clone(),
                            generator=None,
                            solver_state=None,
                            guidance_state=None,
                        ),
                        batch=make_request_batch(initial),
                        negative_batch=None,
                    )
                    events = list(execute(RecordingModel(), [run], sampler.guidance))
                    self.assertEqual(len(events), len(plan))
                    self.assertTrue(torch.isfinite(run.ctx.latents).all())
                    results.append(run.ctx.latents)
                assert_bitwise(results[0], results[1])

    def test_sa_window_gating_composes_deterministic_pec(self) -> None:
        solver = SASolver()  # eta = 0.4 inside [0.3, 0.9]
        sampler = Sampler(steps=6, solver=solver)
        initial = torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(1))

        # eta gated to zero everywhere: deterministic PEC, no generator draws.
        plan = finalize_replay_state(
            with_sde_window(solver.plan(torch.linspace(1.0, 0.0, 7).tolist()), 0, 0)
        )

        def run_gated(seed: int) -> tuple[torch.Tensor, bool]:
            generator = torch.Generator().manual_seed(seed)
            state_before = generator.get_state().clone()
            run = Run(
                plan=plan,
                ctx=StepContext(
                    latents=initial.clone(),
                    generator=generator,
                    solver_state=None,
                    guidance_state=None,
                ),
                batch=make_request_batch(initial),
                negative_batch=None,
            )
            events = list(execute(RecordingModel(), [run], sampler.guidance))
            self.assertEqual(len(events), len(plan))
            return run.ctx.latents, torch.equal(generator.get_state(), state_before)

        first, first_untouched = run_gated(3)
        second, second_untouched = run_gated(19)
        assert_bitwise(first, second)
        self.assertTrue(first_untouched and second_untouched)

        # Sanity: the ungated plan keeps its stored per-transition tau and is
        # actually stochastic (different seeds diverge).
        def sample_full(seed: int) -> torch.Tensor:
            request = SampleRequest(
                batch=make_request_batch(initial),
                generator=torch.Generator().manual_seed(seed),
            )
            model = RecordingModel()
            return sampler.sample(model, [request])[0].final_latents

        self.assertFalse(torch.equal(sample_full(3), sample_full(19)))

    def test_sa_batches_across_requests(self) -> None:
        # The legacy loop sampled SA requests sequentially with a warn_once;
        # the executor now batches every rendezvous across requests.
        sampler = Sampler(steps=4, solver=SASolver(eta=0.0))
        model = microbatching.FakeSamplerModel()
        sampler.sample(
            model,
            [
                SampleRequest(batch=microbatching.make_sampler_batch(1.0)),
                SampleRequest(batch=microbatching.make_sampler_batch(2.0)),
            ],
        )
        # 4 eval rounds (seed + 3 PEC target evals; terminal evaluates nothing),
        # each a single batched forward over both requests.
        self.assertEqual(model.forward_batch_sizes, [2, 2, 2, 2])

    def test_heterogeneous_plans_fall_back_to_grouped_execution(self) -> None:
        # Runs with different plan topologies in one execute() call take the
        # grouped-sequential fallback: same-topology runs still batch together,
        # every run finishes, and results match the homogeneous fast path.
        sampler = Sampler(solver=FlowSolver())
        solver = sampler.solver

        def make_run(steps: int, value: float) -> Run:
            plan = finalize_replay_state(
                solver.plan(torch.linspace(1.0, 0.0, steps + 1).tolist())
            )
            batch = microbatching.make_sampler_batch(value)
            return Run(
                plan=plan,
                ctx=StepContext(
                    latents=batch["noisy_latents"].float(),
                    generator=None,
                    solver_state=None,
                    guidance_state=None,
                ),
                batch=batch,
                negative_batch=None,
            )

        model = microbatching.FakeSamplerModel()
        runs = [make_run(4, 1.0), make_run(6, 2.0), make_run(4, 3.0)]
        events = list(execute(model, runs, sampler.guidance))
        # Groups execute sequentially (4-step group of two runs, then the
        # 6-step run); step indices stay contiguous over the combined total.
        self.assertEqual([event.step_idx for event in events], list(range(10)))
        self.assertTrue(all(event.total_steps == 10 for event in events))
        self.assertEqual(model.forward_batch_sizes, [2, 2, 2, 2, 1, 1, 1, 1, 1, 1])

        for steps, value, run in (
            (4, 1.0, runs[0]),
            (6, 2.0, runs[1]),
            (4, 3.0, runs[2]),
        ):
            reference = make_run(steps, value)
            list(
                execute(microbatching.FakeSamplerModel(), [reference], sampler.guidance)
            )
            assert_bitwise(run.ctx.latents, reference.ctx.latents)


if __name__ == "__main__":
    unittest.main()
