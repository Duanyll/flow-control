"""Endpoint training: the objective contract and the trainer loop over it.

:class:`EndpointTrainer` is the endpoint family (NFT / RAM / AWM / weighted
flow matching) over
:class:`~flow_control.training.rollout_trainer.RolloutTrainerBase`: the clean
rollout endpoint is re-noised at a training timestep and an :class:`Objective`
turns the velocities of the current / old / reference policies there into a
loss. The concrete trainers are presets over ``objective`` and
``train_timesteps``; each method's objective lives next to its preset
(``nft.py`` / ``ram.py`` / ``awm.py``; ``weighted_fm.py`` has no preset).

An :class:`Objective` is the math core of the trainer: given one training
point (a clean endpoint re-noised at a timestep) and the velocities the
candidate policies predict there, it returns a loss and detached metrics. It
holds no device, model or logger; the trainer owns the loop, decides which
policy forwards to run (:meth:`BaseObjective.required_policies`), which policy
samples the rollouts (:meth:`BaseObjective.rollout_policy`), and prefixes the
metrics with ``train/`` before aggregation.

Conventions (flow_control): ``t`` in ``[0, 1]`` with ``1`` = pure noise,
``x_t = (1 - t) * x0 + t * noise`` and the flow-matching target is
``noise - x0``. The reference layering in Signed-RF-Playground flips both
signs; do not port coefficients across without re-deriving.

Forward rule for endpoint items: a *grid* timestep (``grid_index`` set) is
evaluated through the rollout plan's own step
(``SampleRun.guided_velocity`` with ``train_predictor``), so per-step variant
schedules and CFG++ see the executed transition; a *continuous* timestep has no
transition and goes through ``train_predictor.velocity`` directly.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from flow_control.adapters.base import Batch
from flow_control.samplers import Executor, SampleRequest
from flow_control.samplers.calls import Calls
from flow_control.utils.logging import get_logger
from flow_control.utils.registry import Registry, RegistryUnion
from flow_control.utils.tensor import deep_move_to_device

from .ema import EMAConfig, EMAOptimizer, apply_ema_maybe
from .mixins import PendingRollouts, Rollout
from .rollout_trainer import RolloutTrainerBase
from .train_timesteps import TrainTimesteps

logger = get_logger(__name__)


PolicyRole = Literal["old", "ref"]
"""Auxiliary policies an objective may require; ``current`` is always evaluated."""


@dataclass(slots=True)
class TrainPoint:
    """One re-noised endpoint. All tensors are batch-1 and float32."""

    x0: torch.Tensor
    """Clean latents (the rollout endpoint), ``[1, ...]``."""
    noise: torch.Tensor
    """Freshly drawn ``eps`` of the same shape as ``x0``."""
    t: torch.Tensor
    """``[1]`` training timestep, ``1`` = pure noise."""
    advantage: torch.Tensor
    """``[1]`` group-relative advantage of the rollout."""

    @property
    def xt(self) -> torch.Tensor:
        t_expanded = self.t.view(-1, *([1] * (self.x0.ndim - 1)))
        return (1.0 - t_expanded) * self.x0 + t_expanded * self.noise

    @property
    def target(self) -> torch.Tensor:
        return self.noise - self.x0


@dataclass(slots=True)
class PolicyVelocities:
    """Velocities at the same ``(x_t, t)`` from each policy role.

    ``current`` carries gradients; ``old`` (lagged EMA policy) and ``ref``
    (frozen base) come from ``no_grad`` forwards or a cache and are detached by
    every objective, so no gradient ever reaches them.
    """

    current: torch.Tensor
    old: torch.Tensor | None = None
    ref: torch.Tensor | None = None


@dataclass(slots=True)
class LossOutput:
    loss: torch.Tensor
    """0-dim, attached to the graph."""
    metrics: dict[str, torch.Tensor]
    """0-dim detached values without prefix; the trainer adds ``train/``."""


class BaseObjective(BaseModel, ABC):
    type: str
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def rollout_policy(self) -> Literal["current", "old"]:
        """Which policy samples the rollout endpoints."""
        ...

    @abstractmethod
    def required_policies(self) -> frozenset[PolicyRole]:
        """Auxiliary velocities :meth:`compute` needs besides ``current``."""
        ...

    @abstractmethod
    def compute(self, point: TrainPoint, velocities: PolicyVelocities) -> LossOutput:
        """Per-item loss and detached metrics."""
        ...

    def _require(self, velocities: PolicyVelocities, role: PolicyRole) -> torch.Tensor:
        velocity = getattr(velocities, role)
        if velocity is None:
            raise RuntimeError(
                f"Objective {self.type!r} requires the {role!r} policy velocity "
                "but none was provided; check required_policies() against the "
                "trainer's precompute/cache path."
            )
        return velocity.detach()


objective_registry: Registry[BaseObjective] = Registry("objective", base=BaseObjective)


Objective = Annotated[BaseObjective, RegistryUnion(objective_registry, "type")]


POLICY_ROLE_ORDER: tuple[PolicyRole, ...] = ("old", "ref")
"""Auxiliary forwards run in this fixed order on every rank. Iterating the
objective's ``frozenset`` directly would follow per-process string hashing and
could desynchronize the collective forward sequence across ranks."""


@dataclass(slots=True)
class EndpointTrainItem:
    rollout_idx: int
    sigma: float
    """Training timestep, ``1`` = pure noise."""
    grid_index: int | None
    """Index of ``sigma`` on the rollout plan, or ``None`` for a continuous draw."""
    noise: torch.Tensor | None = None
    """The re-noising ``eps``; drawn once by precompute so the cached velocities
    match the training input, otherwise fresh per loss evaluation."""
    cache: dict[PolicyRole, torch.Tensor] = field(default_factory=dict)
    """Detached auxiliary velocities by policy role."""


@dataclass(slots=True)
class _Prepared:
    """One item on device: the row already carries ``noisy_latents = x_t``."""

    item: EndpointTrainItem
    rollout: Rollout
    row: Batch
    negative: Batch | None
    point: TrainPoint


class EndpointTrainer(RolloutTrainerBase[EndpointTrainItem]):
    """Re-noise rollout endpoints and train an :class:`Objective` on them.

    ``objective`` decides which policy samples the rollouts and which auxiliary
    velocities the loss needs; ``train_timesteps`` decides where each endpoint
    is re-noised. The old policy is a lagged EMA (``ema_old``) stepped once per
    outer epoch; the reference policy is the frozen initial weights.
    """

    # The objective members register from the method modules (nft / ram / awm /
    # weighted_fm), which import this class; defer the schema build past import
    # so the ``Objective`` union is not materialized from an empty registry.
    model_config = ConfigDict(extra="forbid", defer_build=True)

    objective: Objective
    train_timesteps: TrainTimesteps
    ema_old: EMAConfig
    """Old / lagged policy EMA (stepped once per outer epoch). Required, like
    ``objective`` and ``train_timesteps``: every preset chooses its own."""
    lookahead_teacher: Literal["latest", "behavior"] = "latest"
    """Which teacher a batch sampled ahead (``rollout_lookahead > 0``) trains
    against. ``latest``: the old EMA as it stands at train time, one lookahead
    step ahead of the policy that sampled the batch. ``behavior``: the old /
    reference velocities are precomputed when the batch is sampled, so the
    teacher equals the behaviour policy by construction; this is the
    ``precompute_aux_model_outputs`` path run early, and both batches' caches
    live on ``rollout_storage_device`` meanwhile. Only meaningful with
    ``rollout_lookahead > 0``. A resume re-primes the batch that was in flight
    and redraws its plan and noise from the checkpoint's RNG state, so a
    ``behavior`` run resumed mid-lookahead is not bitwise identical to an
    uninterrupted one (``latest`` draws them at train time and is)."""

    _old_ema: EMAOptimizer

    @model_validator(mode="after")
    def check_lookahead_teacher(self) -> Self:
        if (
            self.lookahead_teacher == "behavior"
            and not self.precompute_aux_model_outputs
        ):
            raise ValueError(
                "lookahead_teacher='behavior' caches the old / reference velocities "
                "when a batch is sampled, which is the precompute path; set "
                "precompute_aux_model_outputs=true or use lookahead_teacher='latest'."
            )
        return self

    # --------------------------------- Hooks ------------------------------------ #

    def _needs_reference(self) -> bool:
        return "ref" in self.objective.required_policies()

    def _make_aux_optimizers(self, params: list[torch.nn.Parameter]) -> None:
        self._old_ema = EMAOptimizer(params, self.ema_old)
        logger.info(
            f"Old-policy EMA created (decay={self.ema_old.decay}, "
            f"warmup={self.ema_old.warmup.type})."
        )

    def _aux_state_dict(self, opts: StateDictOptions) -> dict[str, Any]:
        return {
            "optim_ema_old": get_optimizer_state_dict(
                self.transformer, self._old_ema, options=opts
            ),
            # DCP keeps only ``state`` / ``param_groups``; the warmup ramp needs
            # the step counter too.
            "ema_old_step_count": self._old_ema.ema_step_count,
        }

    def _load_aux_state_dict(
        self, state_dict: dict[str, Any], opts: StateDictOptions
    ) -> None:
        if "optim_ema_old" in state_dict:
            set_optimizer_state_dict(
                self.transformer,
                self._old_ema,
                state_dict["optim_ema_old"],
                options=opts,
            )
            self._old_ema.coerce_buffer_dtype()
            # Stepped once per outer epoch, so older checkpoints without the
            # counter resume it from the epoch count.
            self._old_ema.ema_step_count = state_dict.get(
                "ema_old_step_count", state_dict.get("current_epoch", 0)
            )

    def _rollout_scope(self) -> AbstractContextManager[None]:
        if self.objective.rollout_policy() == "old":
            return apply_ema_maybe(self._old_ema)
        return nullcontext()

    def _on_rollouts_started(self, pending: PendingRollouts) -> None:
        """``lookahead_teacher="behavior"``: build the batch's train plan and
        cache its teacher velocities now, under the weights that sampled it."""
        if self.lookahead_teacher != "behavior" or self.rollout_lookahead == 0:
            return

        def behavior_scope(role: PolicyRole) -> AbstractContextManager[None]:
            # _rollout_scope() has already applied the old EMA when the old
            # policy samples; apply_ema_maybe is not re-entrant (apply_shadow
            # would back up the shadow itself), so run "old" as the weights are.
            if role == "old" and self.objective.rollout_policy() == "old":
                return nullcontext()
            return self._policy_scope(role)

        train_plan = self._build_train_plan(pending.rollouts)
        self._fill_caches(
            pending.rollouts,
            train_plan,
            torch.zeros(len(pending.rollouts)),
            scope=behavior_scope,
            storage=self.rollout_storage,
        )
        pending.train_plan = train_plan

    def _after_train_epoch(self) -> None:
        self._old_ema.step()

    def _policy_scope(self, role: PolicyRole) -> AbstractContextManager[None]:
        if role == "old":
            return apply_ema_maybe(self._old_ema)
        return self.reference_model()

    # ------------------------------- Train plan --------------------------------- #

    def _build_train_plan(
        self, rollouts: list[Rollout]
    ) -> list[list[EndpointTrainItem]]:
        # Resolve the scalar plan data once per rollout. For device-backed
        # storage this is one host transfer here, never one sync per model eval.
        rollout_sigmas = [
            [transition.sigma for transition in rollout.sampling_plan]
            for rollout in rollouts
        ]
        train_plan: list[list[EndpointTrainItem]] = []
        for _ in range(self.num_inner_epochs):
            items: list[EndpointTrainItem] = []
            for rollout_idx in torch.randperm(len(rollouts)).tolist():
                items.extend(
                    EndpointTrainItem(rollout_idx, sigma, grid_index)
                    for sigma, grid_index in self.train_timesteps.draw(
                        rollout_sigmas[rollout_idx]
                    )
                )
            train_plan.append(items)
        return train_plan

    # ------------------------------- Prediction --------------------------------- #

    def _make_point(
        self, item: EndpointTrainItem, row: Batch, advantage: torch.Tensor
    ) -> TrainPoint:
        x0 = row["clean_latents"].float()
        noise = (
            torch.randn_like(x0)
            if item.noise is None
            else item.noise.to(device=self.device, dtype=torch.float32)
        )
        return TrainPoint(
            x0=x0,
            noise=noise,
            t=torch.tensor([item.sigma], device=self.device, dtype=torch.float32),
            advantage=advantage.to(device=self.device, dtype=torch.float32).view(1),
        )

    def _prepare(
        self,
        items: list[EndpointTrainItem],
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ) -> list[_Prepared]:
        prepared: list[_Prepared] = []
        for item in items:
            rollout = rollouts[item.rollout_idx]
            row = deep_move_to_device(rollout.row, self.device)
            point = self._make_point(item, row, advantages[item.rollout_idx])
            row["noisy_latents"] = point.xt
            prepared.append(
                _Prepared(
                    item,
                    rollout,
                    row,
                    deep_move_to_device(rollout.negative_row, self.device),
                    point,
                )
            )
        return prepared

    def _predict(self, prepared: list[_Prepared]) -> list[torch.Tensor]:
        """``train_predictor`` velocities at every point, in one collective pass.

        One ``train_timesteps`` draws either only grid or only continuous
        timesteps, so a microbatch is homogeneous. Continuous points are
        independent evaluations; grid points run through the rollout plan's
        step so the predictor sees the executed transition.
        """
        if prepared[0].item.grid_index is None:
            return self.predict_training(
                [entry.row for entry in prepared],
                [entry.point.t for entry in prepared],
                [entry.negative for entry in prepared],
            )
        gens: list[Calls[torch.Tensor]] = []
        for entry in prepared:
            row, negative, plan = (
                entry.row,
                entry.negative,
                entry.rollout.sampling_plan,
            )
            run = self.rollout_sampler.make_run(
                SampleRequest(
                    row=row,
                    negative_row=self.training_negative(row, len(plan), negative),
                ),
                plan=plan,
                predictor=self.train_predictor,
            )
            assert entry.item.grid_index is not None, "homogeneous microbatch"
            gens.append(run.guided_velocity(entry.point.xt, entry.item.grid_index))
        # Enumerate the configured grid, not this rank's executed plan: an
        # SDEdit-sliced plan is row dependent, and every rank must run the
        # same variant collectives (same rule as GRPO's replay_steps).
        variants = self.train_predictor.variant_keys(self.rollout_sampler.steps)
        return Executor(self.model, variants or [None]).evaluate(gens)

    def _required_roles(self) -> Iterator[PolicyRole]:
        required = self.objective.required_policies()
        return (role for role in POLICY_ROLE_ORDER if role in required)

    def _loss_batched(
        self,
        items: list[EndpointTrainItem],
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self._prepare(items, rollouts, advantages)
        aux: dict[PolicyRole, list[torch.Tensor]] = {}
        for role in self._required_roles():
            if all(role in entry.item.cache for entry in prepared):
                aux[role] = [
                    entry.item.cache[role].to(device=self.device, dtype=torch.float32)
                    for entry in prepared
                ]
            else:
                with torch.no_grad(), self._policy_scope(role):
                    aux[role] = [v.detach() for v in self._predict(prepared)]

        losses: list[torch.Tensor] = []
        for index, (entry, current) in enumerate(
            zip(prepared, self._predict(prepared), strict=True)
        ):
            out = self.objective.compute(
                entry.point,
                PolicyVelocities(
                    current=current,
                    old=aux["old"][index] if "old" in aux else None,
                    ref=aux["ref"][index] if "ref" in aux else None,
                ),
            )
            self.log_aggregated_metrics(
                {f"train/{key}": value for key, value in out.metrics.items()}
            )
            losses.append(out.loss)
        return torch.stack(losses).mean()

    # ------------------------------- Precompute --------------------------------- #

    def _precompute(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[EndpointTrainItem]],
        advantages: torch.Tensor,
    ) -> None:
        self._fill_caches(
            rollouts,
            train_plan,
            advantages,
            scope=self._policy_scope,
            storage=self.device,
        )

    def _fill_caches(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[EndpointTrainItem]],
        advantages: torch.Tensor,
        scope: Callable[[PolicyRole], AbstractContextManager[None]],
        storage: torch.device,
    ) -> None:
        """Draw every item's noise once and cache the required auxiliary
        velocities at the resulting ``x_t``. *scope* gives the weights each role
        runs under; the cached tensors are kept on *storage* (``_make_point`` /
        ``_loss_batched`` move them back to the device)."""
        flat_items = [item for items in train_plan for item in items]
        if len(flat_items) == 0:
            return

        with self._precompute_scope() as progress:
            prepare_task = progress.add_task("Prepare cache", total=len(flat_items))
            # Draw the noise once so every cached velocity sees the exact x_t
            # the update loop will train on.
            for item in flat_items:
                item.noise = torch.randn_like(
                    rollouts[item.rollout_idx].row["clean_latents"],
                    device=self.device,
                    dtype=torch.float32,
                ).to(storage)
                progress.advance(prepare_task)
            for role in self._required_roles():
                task = progress.add_task(f"Precompute {role}", total=len(flat_items))
                with scope(role):
                    for micro_items in self.iter_train_micro_batches(flat_items):
                        prepared = self._prepare(micro_items, rollouts, advantages)
                        velocities = self._predict(prepared)
                        for item, velocity in zip(micro_items, velocities, strict=True):
                            item.cache[role] = velocity.detach().to(storage)
                        progress.advance(task, advance=len(micro_items))
