"""The RL epoch loop on the real NFT trainer code with the DiT faked: a tiny
LoRA adapter with ``ema`` / ``old`` / ``ref`` variants, the reward backend and
the store stubbed. CPU, world size 1."""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch

import torch
from diffusers import ModelMixin
from pydantic import Field, PrivateAttr
from test_trainer_recipe_config import _NftProbe, _ProbeOverrides
from test_variants import TinyTransformer

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.data import KEY, Index, IndexEntry, RowCursor
from flow_control.rewards.base import BaseJudge, RemoteReward, judge_registry
from flow_control.samplers import Grid, Row
from flow_control.training.mixins import PendingRollouts
from flow_control.training.nft import NftTrainer
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.tensor import deep_apply_tensor_fn

VARIANTS: dict[str, Any] = {
    "default": {"type": "lora", "rank": 2},
    "ema": {"type": "ema", "decay": 0.5},
    "old": {"type": "ema", "decay": 0.9, "every": "epoch"},
    "ref": {"type": "alias", "of": "base"},
}
POINTWISE: dict[str, Any] = {
    "type": "pointwise",
    "sampler": Grid(steps=4),
    "reward": {"type": "_test_key"},
}


@judge_registry.register("_test_tie")
class _TieJudge(BaseJudge, RemoteReward[tuple[Row, Row], float]):
    """Remote judge that calls every pair a tie after a short sleep, so the
    batch's requests are still in flight when the next one samples."""

    type: Literal["_test_tie"] = "_test_tie"

    @property
    def fields(self) -> set[str]:
        return {KEY}

    async def run_one(self, pair: tuple[Row, Row]) -> float:
        row_a, row_b = pair
        assert row_a[KEY] == row_b[KEY]
        await asyncio.sleep(0.005)
        return 0.5


class _Transformer(TinyTransformer):
    def set_requires_gradient_sync(self, flag: bool) -> None:
        """FSDP's hook, called by the update loop."""


class _Adapter(BaseModelAdapter[ModelMixin, Batch]):
    """The tiny two-block DiT; journals the variant of every forward."""

    arch: str = "tiny"
    type: str = "tiny"
    hf_model: HfModelLoader[ModelMixin] = Field(
        default_factory=lambda: HfModelLoader(
            library="custom", class_name="Tiny", pretrained_model_id="tiny"
        )
    )
    lora_target_modules = ("to_q",)
    _forwards: list[str] = PrivateAttr(default_factory=list)

    def _predict_velocity(self, batch: Batch, timestep: torch.Tensor) -> torch.Tensor:
        self._forwards.append(self._active.get())
        return self.module()(batch["noisy_latents"], timestep)


def _load(seed: int = 0) -> _Adapter:
    """Realize ``VARIANTS`` on a fresh transformer the way ``load_transformer`` does."""
    torch.manual_seed(seed)
    adapter = _Adapter(variants=VARIANTS)
    adapter.hf_model.dtype = torch.float32
    adapter.transformer = _Transformer().requires_grad_(False)
    with patch.object(HfModelLoader, "load_model", return_value=True):
        adapter.load_transformer(torch.device("cpu"))
    return adapter


class _RowStore:
    """``n`` prompt rows keyed ``"0" .. "n-1"``; the key doubles as the reward."""

    def __init__(self, n: int) -> None:
        self.index = Index(
            [IndexEntry(str(i), 1, "", None, str(i)) for i in range(n)], {}
        )
        self.gets: list[str] = []
        """Keys fetched, in order: a batch's prompts as the runner pulls them."""

    def __len__(self) -> int:
        return len(self.index)

    def get(self, row_id: int) -> dict[str, Any]:
        self.gets.append(str(row_id))
        return {
            KEY: str(row_id),
            "image_size": (8, 8),
            "clean_latents": torch.zeros(1, 1, 2),
            "noisy_latents": torch.full((1, 1, 2), 0.9),
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

    def _start_rollouts(self, epoch: int) -> PendingRollouts:
        store = cast(_RowStore, self._store)
        first = len(store.gets)
        pending = super()._start_rollouts(epoch)
        self._events.append(("start", pending.epoch, pending.policy_version))
        # sampled() returns once every prompt of the batch was pulled.
        self._sampled[pending.epoch] = [
            key
            for key in store.gets[first:]
            for _ in range(self.num_rollouts_per_prompt)
        ]
        return pending

    def _finish_rollouts(self, pending: PendingRollouts) -> list[Row]:
        self._events.append(("finish", pending.epoch))
        return super()._finish_rollouts(pending)

    def _compute_advantages(self, rows: list[Row], step: int) -> torch.Tensor:
        # World size 1 without a process group: the all_gather is the identity.
        rewards = torch.stack([row["reward"] for row in rows]).float()
        return (rewards - rewards.mean()).squeeze(-1)

    def _train_on_rollouts(self, rows: list[Row], advantages: torch.Tensor) -> None:
        self._events.append(("train", self._current_epoch))
        self._trained.append(
            [(row[KEY], row["reward"].item(), self.rollout.labels) for row in rows]
        )
        super()._train_on_rollouts(rows, advantages)

    def _emit_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._emitted.append((step, dict(metrics)))


def _make_trainer(
    cls: type[Any], config: dict[str, Any], store: _RowStore, model: _Adapter
) -> Any:
    """``run()``'s setup for a CPU probe: seed, fakes, cursor, runner, real optimizer."""
    trainer = cls.model_validate(
        {"train_predictor": "model", "rollout": POINTWISE, **config}
    )
    trainer.set_seed()
    trainer.model = model
    trainer.processor = SimpleNamespace(
        initialize_latents=lambda row, **kwargs: row["noisy_latents"],
        decode_output=lambda latents, row: {},
        resample=lambda row, generator: row,
    )
    trainer._store = store
    trainer._cursor = RowCursor(
        store,
        trainer.make_planner(store, shuffle=True),
        trainer.seed,
        sampling=trainer.prompt_sampling,
    )
    trainer.make_runner(model)
    trainer.make_optimizer_and_scheduler()
    return trainer


def _expected_schedule(lookahead: int, train_epochs: int) -> list[tuple[Any, ...]]:
    """Plan §4.1: prime ``N`` batches under the initial weights; epoch ``k``
    samples ``k + N`` (under the weights after epoch ``k - 1``) before it waits
    for batch ``k`` and trains on it."""
    events: list[tuple[Any, ...]] = [
        ("start", epoch, 0) for epoch in range(min(lookahead, train_epochs))
    ]
    for k in range(train_epochs):
        if k + lookahead < train_epochs:
            events.append(("start", k + lookahead, k))
        events += [("finish", k), ("train", k)]
    return events


def _policy_lags(trainer: _ScheduleProbe) -> list[float]:
    return [
        m["rollout/policy_lag"]
        for _, m in trainer._emitted
        if "rollout/policy_lag" in m
    ]


def _lora(adapter: _Adapter, variant: str) -> dict[str, torch.Tensor]:
    """``blocks.0.to_q``'s A / B of one variant, cloned."""
    pair = cast(Any, adapter.transformer).blocks[0].to_q.lora[variant]
    return {"A": pair.A.weight.detach().clone(), "B": pair.B.weight.detach().clone()}


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
        trainer.close_runner()

    def test_lookahead_schedule_and_resume(self) -> None:
        """The loop must sample batch ``k + N`` before waiting for ``k`` and
        still train on ``k`` with ``k``'s rewards; the checkpoint must rewind the
        prompt cursor past the batch it drops and carry the frozen ``old`` EMA
        with its epoch count. Eight distinct prompts over four epochs make every
        batch identifiable by its keys."""
        for lookahead in (0, 1, 2):
            with self.subTest(lookahead=lookahead):
                trainer = _make_trainer(
                    _ScheduleProbe,
                    {**self.SCHEDULE_CONFIG, "rollout_lookahead": lookahead},
                    _RowStore(8),
                    _load(),
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
                # Every batch logs its sampling time when it is sampled and its
                # reward wait when it trains.
                for key in (
                    "profile/rollout/rollout_elapsed_s",
                    "profile/reward/wait_at_train_s",
                ):
                    self.assertEqual(
                        sum(key in m for _, m in trainer._emitted), self.EPOCHS, key
                    )
                # Epoch k trains on the rollouts sampled for k (not the batch
                # started in the same epoch) and its rewards are already written
                # (a missing reward would mean training before finish, or batch
                # k's rows scored into another batch).
                self.assertEqual(
                    len({key for keys in trainer._sampled.values() for key in keys}),
                    2 * self.EPOCHS,
                )
                for epoch, seen in enumerate(trainer._trained):
                    self.assertEqual(
                        seen,
                        [(key, float(key), ["key"]) for key in trainer._sampled[epoch]],
                    )

        with self.subTest("pairwise program sampled one batch ahead"):
            # A prompt group's pair requests go out as soon as its K rollouts
            # are in, on the run-long loop, so the batch sampled ahead scores
            # while the next one samples and epoch k still trains on k's
            # rewards; the profile counts requests: 2 prompts x 1 pair x both
            # orders.
            trainer = _make_trainer(
                _ScheduleProbe,
                {
                    **self.SCHEDULE_CONFIG,
                    "rollout_lookahead": 1,
                    "num_rollouts_per_prompt": 2,
                    "rollout": {
                        "type": "round_robin",
                        "sampler": Grid(steps=4),
                        "judge": {"type": "_test_tie"},
                    },
                },
                _RowStore(8),
                _load(),
            )
            trainer._prime_rollouts()
            self._run_to_end(trainer)
            self.assertEqual(trainer._events, _expected_schedule(1, self.EPOCHS))
            self.assertEqual(_policy_lags(trainer), [0, 1, 1, 1])
            for epoch, seen in enumerate(trainer._trained):
                self.assertEqual(
                    seen,
                    [(key, 0.5, ["round_robin"]) for key in trainer._sampled[epoch]],
                )
            counts = [
                m["profile/reward/count"]
                for _, m in trainer._emitted
                if "profile/reward/count" in m
            ]
            self.assertEqual(counts, [4.0] * self.EPOCHS)

        with self.subTest("resume with a batch in flight"):
            config = {**self.SCHEDULE_CONFIG, "rollout_lookahead": 1}
            model = _load()
            trainer = _make_trainer(_ScheduleProbe, config, _RowStore(8), model)
            trainer._prime_rollouts()
            for _ in range(2):
                trainer._run_epoch()
            in_flight = trainer._pending_rollouts[0]
            self.assertEqual(in_flight.epoch, 2)
            # A plain module's state dict aliases the live weights; DCP would not.
            state = deep_apply_tensor_fn(trainer.state_dict(), lambda t: t.clone())
            # The cursor already walked past batch 2's prompts; saving it as is
            # would skip those prompts after the resume.
            self.assertEqual(state["cursor"], in_flight.cursor_state_before)
            self.assertNotEqual(state["cursor"], trainer._cursor.state_dict())
            self.assertEqual(state["current_epoch"], 2)
            # The trained LoRA and every frozen variant (EMA slots, counters)
            # are in the model state; the frozen base weights are not.
            weights = state["transformer"]
            self.assertEqual(int(weights["variant_counts.old"]), 2)
            for variant in ("default", "ema", "old"):
                self.assertIn(f"blocks.0.to_q.lora.{variant}.B.weight", weights)
            self.assertNotIn("blocks.0.to_q.weight", weights)

            resumed_model = _load(seed=1)
            resumed = _make_trainer(_ScheduleProbe, config, _RowStore(8), resumed_model)
            resumed.load_state_dict(state)
            self.assertEqual(
                int(cast(Any, resumed_model.transformer).variant_counts.old), 2
            )
            for variant in ("default", "ema", "old"):
                torch.testing.assert_close(
                    _lora(resumed_model, variant), _lora(model, variant), rtol=0, atol=0
                )
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
                    ("finish", 3),
                    ("train", 3),
                ],
            )
            self.assertEqual(_policy_lags(resumed), [0, 1])
            self.assertEqual(
                resumed._sampled, {epoch: trainer._sampled[epoch] for epoch in (2, 3)}
            )

        with self.subTest("resume with the in-flight batch kept"):
            # checkpoint_rollouts saves batch 2 scored: the resume trains on the
            # rows the uninterrupted run trained on, at its lag, without
            # sampling them again; the cursor stays past batch 2's prompts.
            config = {
                **self.SCHEDULE_CONFIG,
                "rollout_lookahead": 1,
                "checkpoint_rollouts": True,
            }
            trainer = _make_trainer(_ScheduleProbe, config, _RowStore(8), _load())
            trainer._prime_rollouts()
            for _ in range(2):
                trainer._run_epoch()
            state = deep_apply_tensor_fn(trainer.state_dict(), lambda t: t.clone())
            self.assertEqual(state["cursor"], trainer._cursor.state_dict())

            resumed = _make_trainer(_ScheduleProbe, config, _RowStore(8), _load(1))
            resumed.load_state_dict(state)
            resumed._prime_rollouts()
            self._run_to_end(trainer)
            self._run_to_end(resumed)
            self.assertEqual(
                resumed._events,
                [
                    ("start", 3, 2),
                    ("finish", 2),
                    ("train", 2),
                    ("finish", 3),
                    ("train", 3),
                ],
            )
            self.assertEqual(_policy_lags(resumed), [1, 1])
            self.assertEqual(resumed._trained, trainer._trained[2:])

    def test_nft_epoch_runs_under_the_declared_variants(self) -> None:
        """One NFT epoch through the real loop: rollouts sample under ``old``,
        each microbatch reads ``old`` and ``ref`` (the base) under no_grad
        before the ``default`` forward, only the LoRA trains, ``ema`` moves on
        the optimizer step and ``old`` at the end of the epoch, and the
        checkpoint carries every variant without the frozen base."""
        model = _load()
        trainer = _make_trainer(
            _NftProbe,
            {
                "num_prompts_per_epoch": 2,
                "num_rollouts_per_prompt": 2,
                "group_size": 4,
                "train_batch_size": 16,
                "train_epochs": 1,
            },
            _RowStore(2),
            model,
        )
        self.addCleanup(trainer.close_runner)
        base = cast(Any, model.transformer).embed.weight.detach().clone()
        before = {name: _lora(model, name) for name in ("default", "ema", "old")}
        self.assertTrue(
            all(torch.equal(w["B"], torch.zeros(4, 2)) for w in before.values())
        )
        with patch.object(
            _NftProbe, "_compute_advantages", _ScheduleProbe._compute_advantages
        ):
            trainer._prime_rollouts()
            trainer._run_epoch()
        # 4 rollouts x 4 steps under "old"; 16 items x (old, ref, default).
        self.assertEqual(model._forwards[:16], ["old"] * 16)
        self.assertEqual(model._forwards[16:], ["old", "base", "default"] * 16)
        self.assertEqual(trainer._current_step, 1)
        after = {name: _lora(model, name) for name in ("default", "ema", "old")}
        torch.testing.assert_close(cast(Any, model.transformer).embed.weight, base)
        trained = after["default"]["B"].float()  # trainable_dtype bf16; EMAs fp32
        self.assertFalse(torch.equal(trained, before["default"]["B"].float()))
        torch.testing.assert_close(after["ema"]["B"], 0.5 * trained)
        torch.testing.assert_close(after["old"]["B"], 0.1 * trained)
        counts = cast(Any, model.transformer).variant_counts
        self.assertEqual((int(counts.ema), int(counts.old)), (1, 1))
        weights = trainer.state_dict()["transformer"]
        self.assertEqual(
            {key for key in weights if key.startswith("blocks.0.to_q")},
            {
                f"blocks.0.to_q.lora.{variant}.{slot}.weight"
                for variant in ("default", "ema", "old")
                for slot in ("A", "B")
            },
        )
        self.assertNotIn("embed.weight", weights)


if __name__ == "__main__":
    unittest.main()
