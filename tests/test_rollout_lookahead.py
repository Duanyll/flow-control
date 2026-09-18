"""``rollout_lookahead`` (N-step-off, local/n-step-off/README.md §4): the RL epoch
loop with batches sampled ahead of training, resume with a batch in flight, and
arm B's behaviour-teacher cache. CPU, world size 1, real NFT trainer code with
only the reward backend, the store and the DiT faked."""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from pydantic import PrivateAttr
from test_microbatching import FakeSamplerModel, make_sampler_batch
from test_trainer_recipe_config import _NftProbe, _ProbeOverrides

from flow_control.data import KEY, Index, IndexEntry, RowCursor
from flow_control.rewards import PendingRewards
from flow_control.rewards.base import BaseReward
from flow_control.training.ema import apply_ema_maybe
from flow_control.training.endpoint import EndpointTrainItem
from flow_control.training.mixins import PendingRollouts, Rollout
from flow_control.training.nft import NftTrainer

SUBMIT_REWARD = "flow_control.training.mixins.rollout.submit_reward"


class _KeyReward(BaseReward):
    """Remote-like reward: overlaps with sampling, scores ``float(row[KEY])``
    after a short sleep and records the event loop every request ran on."""

    type: str = "key"
    _loops: list[asyncio.AbstractEventLoop] = PrivateAttr(default_factory=list)

    @property
    def _row_fields(self) -> set[str]:
        return {KEY}

    def _load_model(self, device: torch.device) -> None:
        pass

    def _score(self, row: dict[str, Any]) -> torch.Tensor:
        return torch.tensor([float(row[KEY])])

    async def _async_score(self, row: dict[str, Any]) -> torch.Tensor:
        self._loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0.005)
        return self._score(row)

    def supports_rollout_overlap(self) -> bool:
        return True


class _GainTransformer(torch.nn.Module):
    """One trainable scalar, so optimizer, EMAs and reference backup are real."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def set_requires_gradient_sync(self, flag: bool) -> None:
        """FSDP's hook, called by the update loop."""


class _GainModel(FakeSamplerModel):
    peft_lora_rank = 0

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _GainTransformer()

    @property
    def gain(self) -> torch.nn.Parameter:
        return self.transformer.gain

    def predict_velocity_batched(self, batches, timesteps, *, dummy_outputs=None):
        self.forward_batch_sizes.append(len(batches))
        return [
            self.gain * (b["prompt_embeds"] + 0.1 * b["noisy_latents"]) for b in batches
        ]


class _RowStore:
    """``n`` prompt rows keyed ``"0" .. "n-1"``; the key doubles as the reward."""

    def __init__(self, n: int) -> None:
        self.index = Index(
            [IndexEntry(str(i), 1, "", None, str(i)) for i in range(n)], {}
        )

    def __len__(self) -> int:
        return len(self.index)

    def get(self, row_id: int) -> dict[str, Any]:
        return {
            **make_sampler_batch(0.3, 0.9),
            KEY: str(row_id),
            "prompt_embeds": torch.tensor([[[float(row_id)]]]),
        }


class _ScheduleProbe(_ProbeOverrides, NftTrainer):
    """NFT trainer that journals the epoch loop and captures emitted metrics."""

    _events: list[tuple[Any, ...]] = PrivateAttr(default_factory=list)
    _sampled: dict[int, list[str]] = PrivateAttr(default_factory=dict)
    """epoch -> prompt keys of the batch sampled for it."""
    _trained: list[list[tuple[str, float, list[str]]]] = PrivateAttr(
        default_factory=list
    )
    """Per training epoch: ``(key, reward, labels)`` of every rollout as the
    update loop saw them."""
    _emitted: list[tuple[int, dict[str, float]]] = PrivateAttr(default_factory=list)

    def _on_rollouts_started(self, pending: PendingRollouts) -> None:
        self._events.append(("start", pending.epoch, pending.policy_version))
        self._sampled[pending.epoch] = [r.key for r in pending.rollouts]
        super()._on_rollouts_started(pending)

    def _finish_rollouts(self, pending: PendingRollouts) -> list[Rollout]:
        self._events.append(("finish", pending.epoch))
        return super()._finish_rollouts(pending)

    def _compute_advantages(self, rollouts: list[Rollout], step: int) -> torch.Tensor:
        # World size 1 without a process group: the all_gather is the identity.
        rewards = torch.stack([r.reward for r in rollouts]).float()
        return (rewards - rewards.mean()).squeeze(-1)

    def _train_on_rollouts(
        self,
        rollouts: list[Rollout],
        advantages: torch.Tensor,
        train_plan: list[list[EndpointTrainItem]] | None = None,
    ) -> None:
        self._events.append(("train", self._current_epoch))
        self._trained.append(
            [(r.key, r.reward.item(), list(r.reward_labels)) for r in rollouts]
        )
        super()._train_on_rollouts(rollouts, advantages, train_plan=train_plan)

    def _after_train_epoch(self) -> None:
        self._events.append(("ema_step",))
        super()._after_train_epoch()

    def _emit_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._emitted.append((step, dict(metrics)))


def _make_trainer(
    cls: type[Any],
    config: dict[str, Any],
    store: _RowStore,
    model: _GainModel,
    reward: _KeyReward | None = None,
) -> Any:
    """``run()``'s setup for a CPU probe: seed, fakes, cursor, real optimizers."""
    trainer = cls.model_validate(
        {"train_predictor": "model", "rollout_sampler": {"steps": 4}, **config}
    )
    trainer.set_seed()
    trainer.model = model
    trainer.processor = SimpleNamespace(
        initialize_latents=lambda row, **kwargs: None,
        decode_output=lambda latents, row: {},
        get_negative_row=lambda row: None,
        resample=lambda row, generator: row,
    )
    trainer.reward = reward
    trainer._store = store
    trainer._cursor = RowCursor(
        store,
        trainer.make_planner(store, shuffle=True),
        trainer.seed,
        sampling=trainer.prompt_sampling,
    )
    trainer.make_optimizer_and_scheduler()
    return trainer


def _drain(reward, submitter, loop, profile) -> PendingRewards[int]:
    list(submitter)
    return PendingRewards([])


def _expected_schedule(lookahead: int, train_epochs: int) -> list[tuple[Any, ...]]:
    """Plan §4.1: prime ``N`` batches under the initial weights; epoch ``k``
    samples ``k + N`` (under the weights after epoch ``k - 1``) before it waits
    for batch ``k``, trains on it and steps the old EMA."""
    events: list[tuple[Any, ...]] = [
        ("start", epoch, 0) for epoch in range(min(lookahead, train_epochs))
    ]
    for k in range(train_epochs):
        if k + lookahead < train_epochs:
            events.append(("start", k + lookahead, k))
        events += [("finish", k), ("train", k), ("ema_step",)]
    return events


def _policy_lags(trainer: _ScheduleProbe) -> list[float]:
    return [
        m["rollout/policy_lag"]
        for _, m in trainer._emitted
        if "rollout/policy_lag" in m
    ]


class RolloutLookaheadTest(unittest.TestCase):
    EPOCHS = 4
    SCHEDULE_CONFIG: dict[str, Any] = {
        "num_prompts_per_epoch": 2,
        "group_size": 4,
        "train_epochs": EPOCHS,
    }

    def _run_to_end(self, trainer: _ScheduleProbe) -> None:
        while trainer._current_epoch < trainer.train_epochs:
            trainer._run_epoch()
        trainer.close_reward_loop()

    def test_lookahead_schedule_and_resume(self) -> None:
        """The loop must sample batch ``k + N`` before waiting for ``k`` and
        still train on ``k`` with ``k``'s rewards; the checkpoint must rewind the
        prompt cursor past the batch it drops. Eight distinct prompts over four
        epochs make every batch identifiable by its keys."""
        for lookahead in (0, 1, 2):
            with self.subTest(lookahead=lookahead):
                reward = _KeyReward()
                trainer = _make_trainer(
                    _ScheduleProbe,
                    {**self.SCHEDULE_CONFIG, "rollout_lookahead": lookahead},
                    _RowStore(8),
                    _GainModel(),
                    reward,
                )
                trainer._prime_rollouts()
                self._run_to_end(trainer)

                self.assertEqual(
                    trainer._events, _expected_schedule(lookahead, self.EPOCHS)
                )
                # Primed batches were sampled at epoch 0, later ones N epochs
                # before they train; a wrong policy_version misreports staleness.
                self.assertEqual(
                    _policy_lags(trainer),
                    [min(lookahead, epoch) for epoch in range(self.EPOCHS)],
                )
                waits = [
                    m["profile/reward/wait_at_train_s"]
                    for _, m in trainer._emitted
                    if "profile/reward/wait_at_train_s" in m
                ]
                self.assertEqual(len(waits), self.EPOCHS)
                self.assertEqual(
                    sum("profile/reward/count" in m for _, m in trainer._emitted),
                    self.EPOCHS,
                )
                # Epoch k trains on the rollouts sampled for k (not the batch
                # started in the same epoch) and its rewards are already written
                # (placeholder zeros / ["reward"] would mean training before
                # finish, or batch k's futures written into another batch).
                self.assertEqual(
                    len({key for keys in trainer._sampled.values() for key in keys}),
                    2 * self.EPOCHS,
                )
                for epoch, seen in enumerate(trainer._trained):
                    self.assertEqual(
                        seen,
                        [(key, float(key), ["key"]) for key in trainer._sampled[epoch]],
                    )
                # One event loop for the whole run (a remote client drops its
                # connection pool when the loop changes, cutting off requests in
                # flight), closed once the loop is over.
                loops = reward._loops
                self.assertEqual(len(loops), 2 * self.EPOCHS)
                self.assertTrue(all(loop is loops[0] for loop in loops))
                self.assertTrue(loops[0].is_closed())
                self.assertIsNone(trainer._reward_loop)

        with self.subTest("resume with a batch in flight"):
            config = {**self.SCHEDULE_CONFIG, "rollout_lookahead": 1}
            trainer = _make_trainer(
                _ScheduleProbe, config, _RowStore(8), _GainModel(), _KeyReward()
            )
            trainer._prime_rollouts()
            for _ in range(2):
                trainer._run_epoch()
            in_flight = trainer._pending_rollouts[0]
            self.assertEqual(in_flight.epoch, 2)
            state = trainer.state_dict()
            # The cursor already walked past batch 2's prompts; saving it as is
            # would skip those prompts after the resume.
            self.assertEqual(state["cursor"], in_flight.cursor_state_before)
            self.assertNotEqual(state["cursor"], trainer._cursor.state_dict())
            self.assertEqual(state["current_epoch"], 2)

            resumed = _make_trainer(
                _ScheduleProbe, config, _RowStore(8), _GainModel(), _KeyReward()
            )
            resumed.load_state_dict(state)
            resumed._prime_rollouts()
            self._run_to_end(trainer)
            self._run_to_end(resumed)
            # The dropped batch is re-primed as epoch 2 (lag 0, then N again)
            # and both runs draw the same prompts from the resume point on.
            self.assertEqual(
                resumed._events,
                [
                    ("start", 2, 2),
                    ("start", 3, 2),
                    ("finish", 2),
                    ("train", 2),
                    ("ema_step",),
                    ("finish", 3),
                    ("train", 3),
                    ("ema_step",),
                ],
            )
            self.assertEqual(_policy_lags(resumed), [0, 1])
            self.assertEqual(
                resumed._sampled, {epoch: trainer._sampled[epoch] for epoch in (2, 3)}
            )

    def test_behavior_teacher_cache_matches_sampling_weights(self) -> None:
        """Arm B (plan §4.3): with ``lookahead_teacher="behavior"`` the in-flight
        batch's old / ref velocities are cached inside the rollout scope, so they
        must equal a forward under the weights that sampled the batch even after
        the policy and the old EMA moved on; and running "old" inside the already
        applied EMA shadow must leave the live weights intact on exit."""
        base: dict[str, Any] = {
            "num_inner_epochs": 2,
            "precompute_aux_model_outputs": True,
            "ema_old": {"decay": 0.0, "warmup": {"type": "none"}},
        }
        model = _GainModel()
        with torch.no_grad():
            model.gain.fill_(0.5)
        trainer = _make_trainer(
            _NftProbe,
            {**base, "lookahead_teacher": "behavior", "rollout_lookahead": 1},
            _RowStore(1),
            model,
        )
        self.addCleanup(trainer.close_reward_loop)
        # reference (init backup) = 0.5, behaviour (old EMA) = 1.0, live = 2.0
        with torch.no_grad():
            model.gain.fill_(1.0)
        trainer._old_ema.step()
        with torch.no_grad():
            model.gain.fill_(2.0)

        with patch(SUBMIT_REWARD, side_effect=_drain):
            trainer._launch_rollouts(0)
        # A nested apply_shadow would back up the shadow itself and leave the
        # live weights at the EMA value once the rollout scope exits.
        self.assertEqual(model.gain.item(), 2.0)
        pending = trainer._pending_rollouts[0]
        assert pending.train_plan is not None
        items = [item for epoch in pending.train_plan for item in epoch]
        self.assertEqual(len(items), 4 * 2)  # grid steps x inner epochs
        for item in items:
            assert item.noise is not None
            self.assertEqual(item.noise.device.type, "cpu")
            self.assertEqual(set(item.cache), {"old", "ref"})
            self.assertEqual({v.device.type for v in item.cache.values()}, {"cpu"})

        def forward() -> list[torch.Tensor]:
            with torch.no_grad():
                return trainer._predict(
                    trainer._prepare(items, pending.rollouts, torch.zeros(1))
                )

        with apply_ema_maybe(trainer._old_ema):
            behaviour = forward()
        with trainer.reference_model():
            reference = forward()
        for item, old, ref in zip(items, behaviour, reference, strict=True):
            torch.testing.assert_close(item.cache["old"], old, rtol=0, atol=0)
            torch.testing.assert_close(item.cache["ref"], ref, rtol=0, atol=0)

        # Epoch k's update and the old-EMA step after it: the "latest" teacher
        # now differs from the one that sampled the batch.
        with torch.no_grad():
            model.gain.fill_(3.0)
        trainer._after_train_epoch()
        with apply_ema_maybe(trainer._old_ema):
            latest = forward()
        for item, new in zip(items, latest, strict=True):
            self.assertFalse(torch.allclose(item.cache["old"], new))

        # Training on the prebuilt plan neither rebuilds it nor precomputes
        # again: exactly one (current-policy) forward per item.
        model.forward_batch_sizes.clear()
        with (
            patch.object(_NftProbe, "_build_train_plan", side_effect=AssertionError),
            patch.object(_NftProbe, "_precompute", side_effect=AssertionError),
        ):
            trainer._train_on_rollouts(
                pending.rollouts, torch.zeros(1), train_plan=pending.train_plan
            )
        self.assertEqual(len(model.forward_batch_sizes), len(items))

        for control in (
            {"lookahead_teacher": "latest", "rollout_lookahead": 1},
            {"lookahead_teacher": "behavior", "rollout_lookahead": 0},
        ):
            with self.subTest(**control):
                other = _make_trainer(
                    _NftProbe, {**base, **control}, _RowStore(1), _GainModel()
                )
                self.addCleanup(other.close_reward_loop)
                with patch(SUBMIT_REWARD, side_effect=_drain):
                    other._launch_rollouts(0)
                self.assertIsNone(other._pending_rollouts[0].train_plan)
        with self.assertRaisesRegex(ValueError, "precompute_aux_model_outputs"):
            _NftProbe.model_validate(
                {"train_predictor": "model", "lookahead_teacher": "behavior"}
            )


if __name__ == "__main__":
    unittest.main()
