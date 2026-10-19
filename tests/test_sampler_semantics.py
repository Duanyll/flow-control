"""Node-tree sampler semantics: fixture parity, executor rounds, GRPO record/replay.

* **Parity** (``BaselineParityTest``) replays the numeric fixtures under
  ``tests/fixtures/sampler_baselines/`` — first captured from the pre
  plan-as-data implementation — so accidental numeric drift in any solver
  surfaces as a bitwise mismatch. Regenerate with
  ``tests/sampler_baseline_capture.py`` only when behaviour changes on purpose.
* **Invariants** compare two executions of the current code; no fixtures.
"""

import copy
import importlib
import sys
import unittest
from pathlib import Path

import torch

from flow_control.samplers import (
    BaseGrid,
    BaseTransition,
    Cfg,
    Cps,
    Dance,
    Ddim,
    Euler,
    Executor,
    Flash,
    Grid,
    InitLatent,
    Renoise,
    SaGrid,
    SdeWindowGrid,
    Sequential,
    Tiled,
    select_sde_window,
)
from flow_control.utils.tensor import deep_apply_tensor_fn

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
capture = importlib.import_module("sampler_baseline_capture")
microbatching = importlib.import_module("test_microbatching")


def load_fixture(name: str) -> dict:
    return torch.load(capture.BASELINE_DIR / name, weights_only=True)


def assert_bitwise(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=0, atol=0)


class BaselineParityTest(unittest.TestCase):
    """Bitwise parity against the checked-in numeric baselines.

    A failure here means a solver's numerics moved. If that was intentional,
    regenerate the fixtures (see ``tests/sampler_baseline_capture.py``) and
    commit them with the change; if not, it is the regression these exist for.
    """

    def test_e2e_trajectories_and_eval_topology_match_baselines(self) -> None:
        for name, sampler in capture.e2e_configs().items():
            with self.subTest(fixture=name):
                fixture = load_fixture(f"e2e_{name}.pt")
                actual, model = capture.e2e_fixture(sampler)
                assert_bitwise(actual["final_latents"], fixture["final_latents"])
                # Where and in what order the model is queried is a property of
                # the tree (warmup evals, PEC target points), checked on its own.
                self.assertEqual(model.eval_sigmas, fixture["eval_sigmas"])
                for a, e in zip(
                    model.eval_latents, fixture["eval_latents"], strict=True
                ):
                    assert_bitwise(a, e)
                if actual["traj_latents"] is None:
                    # Multistep grids run no transitions; their path is pinned
                    # by the evals. SA's legacy capture reported the raw grid
                    # head rather than its initial_time cap.
                    self.assertIsInstance(sampler, BaseGrid)
                    self.assertNotIsInstance(sampler, Grid)
                    assert_bitwise(actual["timesteps"][1:], fixture["timesteps"][1:])
                else:
                    assert_bitwise(actual["timesteps"], fixture["timesteps"])
                    assert_bitwise(actual["traj_latents"], fixture["traj_latents"])
                    assert_bitwise(actual["log_probs"], fixture["log_probs"])

    def test_transition_steps_match_stochastic_step_baselines(self) -> None:
        # Next latents, likelihood, moments and the replay density of every
        # stochastic transition, including Flash's grid-position noise ramp.
        configs = {**capture.step_configs(), "flash_eta1": capture.flash_transition()}
        for name, transition in configs.items():
            with self.subTest(fixture=name):
                fixture = load_fixture(f"step_{name}.pt")
                fixture.pop("solver_config")
                actual = capture.step_fixture(
                    transition, range(len(fixture["entries"]))
                )
                self.assertEqual(capture.mismatches(actual, fixture, name), [])

    def test_sa_grid_and_internal_updates_match_baselines(self) -> None:
        fixture = load_fixture("sa_internals.pt")
        fixture.pop("solver_config")
        self.assertEqual(capture.mismatches(capture.sa_fixture(), fixture, "sa"), [])
        # SA does not run on the grid it is handed: the head is capped at
        # initial_time and the penultimate time halved (both invisible on a
        # uniform grid, hence the uneven custom grid); tau follows the target
        # time and is zero below the stochasticity band.
        solver = SaGrid(steps=4, custom_sigmas=[1.0, 0.9, 0.5, 0.2, 0.0])
        row = capture.make_row(torch.zeros(1, 4, 2))
        self.assertEqual(solver.sigmas(row), [solver.initial_time, 0.9, 0.5, 0.25, 0.0])
        self.assertEqual([solver.tau(t) for t in (0.9, 0.5, 0.25)], [0.4, 0.4, 0.0])


class ExecutorSemanticsTest(unittest.TestCase):
    def test_deterministic_grids_consume_no_generator_draws(self) -> None:
        # A deterministic DDIM used to draw (and discard) noise; SA with eta=0
        # must not touch the RNG either, so seeds stay comparable across runs.
        for sampler in (Grid(steps=3, transition=Ddim()), SaGrid(steps=4, eta=0.0)):
            with self.subTest(sampler=sampler.type):
                generator = torch.Generator().manual_seed(5)
                state = generator.get_state().clone()
                row = microbatching.sampling_row(
                    microbatching.make_sampler_batch(0.5), generator
                )
                microbatching.sample(microbatching.FakeSamplerModel(), sampler, [row])
                self.assertTrue(torch.equal(generator.get_state(), state))

    def test_observers_record_the_window_without_changing_the_path(self) -> None:
        # Recording must not change the sampled path (same RNG consumption).
        # The window is drawn from the row generator before any sampling
        # randomness; its steps run the eta copy of the transition, records
        # chain latent_next -> latent_t, and snapshots carry no tensors.
        sampler = SdeWindowGrid(steps=6, size=2, window_range=(1, 5), eta=0.7)

        def run(observers):
            row = microbatching.sampling_row(
                microbatching.make_sampler_batch(0.4),
                torch.Generator().manual_seed(11),
                observers,
            )
            return microbatching.sample(
                microbatching.FakeSamplerModel(), sampler, [row]
            )[0]

        recorded, plain = run([(BaseTransition, capture.record)]), run([])
        assert_bitwise(recorded["noisy_latents"], plain["noisy_latents"])
        self.assertNotIn("trajectory", plain)
        start, end = select_sde_window(6, 2, (1, 5), torch.Generator().manual_seed(11))
        trajectory = recorded["trajectory"]
        points = capture.stochastic_points(recorded)
        self.assertEqual(len(trajectory), 6)
        self.assertEqual(
            [p["_frame"].parent.index for p in points], list(range(start, end))
        )
        self.assertTrue(all((p["log_prob"] != 0).all() for p in points))
        self.assertTrue(
            all((p["log_prob"] == 0).all() for p in trajectory if p not in points)
        )
        assert_bitwise(trajectory[0]["next_latents"], trajectory[1]["noisy_latents"])
        snapshot = points[0]["_frame"]
        self.assertIs(snapshot.node, sampler._sde)
        self.assertEqual(
            (snapshot.node.eta, snapshot.latents, snapshot.velocity), (0.7, None, None)
        )
        self.assertIs(snapshot.root, recorded["_frame"])
        self.assertEqual(len(snapshot.parent.sigmas), 7)

    def test_rollout_and_replay_are_self_consistent(self) -> None:
        # GRPO-shaped parity: replay every recorded stochastic step through the
        # training path (node.prediction + node.log_prob on a fresh point). The
        # deterministic fake model reproduces the rollout velocity, so the
        # replayed log-prob must match bitwise; the executed node (and its eta)
        # comes from the recorded chain, never from a live sampler config.
        velocity = torch.full((1, 4, 2), 0.3)
        transitions = [
            Euler(eta=0.5),
            Ddim(eta=0.5),
            Cps(eta=0.5),
            Dance(eta=0.5),
            Flash(noise_scale_start=0.9, noise_scale_end=0.7),
        ]
        for transition in transitions:
            with self.subTest(transition=transition.type):
                model = capture.ConstVelocityModel(velocity)
                row = microbatching.sampling_row(
                    microbatching.make_batch(0.9),
                    torch.Generator().manual_seed(3),
                    [(BaseTransition, capture.record)],
                )
                (rollout,) = microbatching.sample(
                    model, Grid(steps=4, transition=transition), [row]
                )
                points = capture.stochastic_points(rollout)
                self.assertEqual(len(points), 3)
                self.assertTrue(all((p["log_prob"] != 0).all() for p in points))
                replayed = Executor(model).evaluate(
                    [capture.replay(rollout, p) for p in points]
                )
                for point, output in zip(points, replayed, strict=True):
                    assert_bitwise(output.log_prob, point["log_prob"])

                # S2 precision review: bf16 storage made replay coefficients
                # bf16 even with an fp32 velocity (mean drifted by 1.17e-4).
                # Compare the SAME stored values, promoted before arithmetic.
                stored = [
                    {
                        **p,
                        "noisy_latents": p["noisy_latents"].bfloat16(),
                        "next_latents": p["next_latents"].bfloat16(),
                    }
                    for p in points
                ]
                promoted = [
                    {
                        **p,
                        "noisy_latents": p["noisy_latents"].float(),
                        "next_latents": p["next_latents"].float(),
                    }
                    for p in stored
                ]
                predictions = [velocity.clone().requires_grad_() for _ in range(2)]
                low, reference = [
                    Executor(capture.ConstVelocityModel(prediction)).evaluate(
                        [capture.replay(rollout, p) for p in records]
                    )
                    for prediction, records in zip(
                        predictions, (stored, promoted), strict=True
                    )
                ]
                for actual, expected in zip(low, reference, strict=True):
                    for name in ("log_prob", "mean", "std_dev"):
                        self.assertEqual(getattr(actual, name).dtype, torch.float32)
                        assert_bitwise(getattr(actual, name), getattr(expected, name))
                for outputs in (low, reference):
                    torch.stack(
                        [output.log_prob.sum() for output in outputs]
                    ).sum().backward()
                torch.testing.assert_close(
                    predictions[0].grad, predictions[1].grad, rtol=0, atol=0
                )

    def test_flash_replay_reads_the_grid_position(self) -> None:
        # Flash's noise scale ramps with the step index: a windowed rollout
        # starts recording at index > 0 and the snapshot chain must carry the
        # executed position (a shifted index changes the likelihood).
        velocity = torch.full((1, 4, 2), 0.3)
        model = capture.ConstVelocityModel(velocity)
        sampler = SdeWindowGrid(
            steps=8,
            size=3,
            window_range=(2, 7),
            eta=1.0,
            transition=Flash(eta=0.0, noise_scale_start=0.9, noise_scale_end=0.7),
        )
        row = microbatching.sampling_row(
            microbatching.make_batch(0.9),
            torch.Generator().manual_seed(3),
            [(BaseTransition, capture.record)],
        )
        (rollout,) = microbatching.sample(model, sampler, [row])
        start, end = select_sde_window(8, 3, (2, 7), torch.Generator().manual_seed(3))
        points = capture.stochastic_points(rollout)
        self.assertGreater(start, 0)
        self.assertEqual(
            [p["_frame"].parent.index for p in points], list(range(start, end))
        )
        replayed = Executor(model).evaluate(
            [capture.replay(rollout, p) for p in points]
        )
        for point, output in zip(points, replayed, strict=True):
            assert_bitwise(output.log_prob, point["log_prob"])
        shifted = copy.copy(points[0]["_frame"])
        shifted.parent = copy.copy(shifted.parent)
        shifted.parent.index = 0
        (wrong,) = Executor(model).evaluate(
            [capture.replay(rollout, {**points[0], "_frame": shifted})]
        )
        self.assertFalse(torch.equal(wrong.log_prob, points[0]["log_prob"]))

    def test_sa_batches_across_requests(self) -> None:
        # The legacy loop sampled SA requests sequentially with a warn_once;
        # the executor batches every rendezvous across requests.
        model = microbatching.FakeSamplerModel()
        rows = [
            microbatching.sampling_row(microbatching.make_sampler_batch(v))
            for v in (1.0, 2.0)
        ]
        microbatching.sample(model, SaGrid(steps=4, eta=0.0), rows)
        # 4 eval rounds (seed + 3 PEC target evals; terminal evaluates nothing),
        # each a single batched forward over both requests.
        self.assertEqual(model.forward_batch_sizes, [2, 2, 2, 2])

    def test_progress_counts_every_model_call(self) -> None:
        # Serving reports progress once per ModelCall through RootFrame.on_call,
        # so CFG and tile branches count and estimate_calls must agree with it
        # (init_latent / renoise make no calls).
        class Leaf(microbatching.FakeSamplerModel):
            def predict_velocity_batched(
                self, batches, timesteps, *, dummy_outputs=None
            ):
                self.forward_batch_sizes.append(len(batches))
                return [2 * batch["noisy_latents"] for batch in batches]

        cfg = Cfg(scale=2.0, prediction=Tiled())
        sampler = Sequential(
            steps=[
                InitLatent(source="clean", sigma_end=0.5),
                Grid(steps=3, sigma_end=0.2, transition=Euler(prediction=cfg)),
                Renoise(sigma_end=0.3),
                Grid(steps=2, transition=Euler(prediction=Tiled())),
            ]
        )
        plain = {**microbatching.make_batch(tokens=4), "negative": {}}
        tiled = {
            **plain,
            "image_size": (6, 6),
            "clean_latents": torch.zeros(1, 36, 1),
            "noisy_latents": torch.arange(36, dtype=torch.float32).reshape(1, 36, 1),
            "tiling": {"tile_size": 4, "overlap": 2, "stride": 1},
        }
        for batch, expected in ((plain, 3 * 2 + 2), (tiled, 3 * 2 * 4 + 2 * 4)):
            with self.subTest(tiles="tiling" in batch):
                calls = []
                row = microbatching.sampling_row(
                    batch, torch.Generator().manual_seed(0), on_call=calls.append
                )
                self.assertEqual(sampler.estimate_calls(row), expected)
                microbatching.sample(Leaf(), sampler, [row])
                self.assertEqual(len(calls), expected)
                self.assertEqual(row["sigma"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
