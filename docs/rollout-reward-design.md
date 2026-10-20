# Rollout programs and rewards

A rollout program (`rollout.type`) turns one prompt into its K scored rollouts: it samples (the policy rollouts, and whatever else it wants from the model), calls rewards, and assembles the reward components the advantage sees. Core knows two things about a reward:

- **Placement**: where a call executes. `LocalReward` runs on the device worker (main thread), batched across rows and groups; `RemoteReward` runs on the reward loop (background thread), one request per item, concurrently. Placement alone decides scheduling: a reward's inputs and outputs are its own, and `ctx.run(reward, items)` is the one way to call any of them.
- **Slot**: the contract of a config field. Two are in tree. A `Scorer` takes a row and returns `[C]` raw on its labels; it carries `name` / `weight` / `normalize` and is the only kind the framework aggregates and logs, so `reward`, `validation_scorer` and inference `reward` accept exactly scorers. A `Judge` takes `(a, b)` and returns P(a ≻ b), 0.5 a tie; `round_robin.judge` accepts exactly judges. A program turns judge results into its own components and owns their weights. An out-of-tree program declares fields of either slot, or brings its own registry.

| Slot ↓ / Placement → | `LocalReward` | `RemoteReward` |
|---|---|---|
| `Scorer` | `pickscore`, `hpsv2`, `clip_score`, `clip_image_similarity`, `aesthetic`, `image_reward`, `ocr`, `geneval`; contrib `pyiqa` | `rational_rewards_t2i`, `rational_rewards_edit`, `unified_reward`, `remote` |
| `Judge` | out of tree (a preference network) | out of tree (MLLM pair judges) |

A program awaits every reward the same way, whatever its placement, as the executor sees every sampler node as a `Calls` generator whatever its layer.

A program has two phases. `sample` makes every model call of its group (and may call rewards, e.g. to resample). `score` calls rewards only. The epoch loop trains once every group of a batch has left `sample`.

## Config Specs Examples

### Pointwise

```jsonc
{
  "num_prompts_per_epoch": 48, // a prompt's K rollouts stay on one rank: divisible by world_size
  "num_rollouts_per_prompt": 24,
  "rollout_lookahead": 1, // batches sampled ahead of training (n-step-off)
  "rollout": {
    "type": "pointwise", // every rollout scored on its own by `reward`
    "sampler": {"type": "grid", "steps": 10, "transition": {"type": "euler", "eta": 0.9}},
    "reward": {"type": "pickscore"} // shorthand for a one-item list
  }
}
```

### Several Scorers

```jsonc
"reward": [
  {"type": "pickscore"},                                // its class default normalize: affine 1/26
  {"type": "hpsv2", "weight": 0.5, "normalize": {"type": "clamp", "low": 0.0, "high": 0.4}},
  {"type": "clip_score", "name": "clip", "weight": 0.2} // name: the label prefix (default: type)
]
// labels: pickscore, hpsv2, clip. A multi-component scorer logs "<name>/<label>".
```

### MLLM Scorer

```jsonc
"reward": {
  "type": "rational_rewards_t2i",
  "llm": {"base_url": "http://127.0.0.1:8000/v1", "model": "RationalRewards-8B-T2I", "max_concurrency": 8}
}
// labels: rational_rewards_t2i/overall (weight 1); rational_rewards_t2i/text_faithfulness,
// .../physical_quality, .../text_rendering (weight 0, logged)
```

### Remote Scorer

The server owns its scorers' configs; the client names where and which.

```jsonc
// flow-control reward-server scorers.jsonc
{
  "imports": [],
  "port": 8001,
  "scorers": {"pickscore": {"type": "pickscore"}, "hps": {"type": "hpsv2"}} // local scorers
}
```

```jsonc
"reward": [
  {"type": "remote", "url": "http://node-12:8001", "scorer": "pickscore",
   "normalize": {"type": "affine", "scale": 0.0385}}, // the server returns raw; weight and normalize are the client's
  {"type": "hpsv2"}
]
```

### Pairwise (Pref-GRPO)

```jsonc
{
  "imports": ["my_plugins.judges"], // registers the judge "my_pair_judge"
  "rollout": {
    "type": "round_robin",
    "sampler": {...},
    "judge": {"type": "my_pair_judge", "llm": {...}},
    // "swap_orders": true, // ask (a, b) and (b, a): s = (s_ab + 1 - s_ba) / 2
    "opponents": 4, // even d: d opponents per row along a random cycle; null: all K(K-1)/2 pairs
    "weight": 0.5,  // of the win-rate component "round_robin"
    "reward": [{"type": "hpsv2", "weight": 0.5}] // scorers on every rollout, alongside
  }
  // validation_scorer defaults to the program's `reward` (hpsv2 here); with none, name one or set false
}
```

### Out-of-Tree: Judge Against an Online EMA Anchor

The anchor is sampled in the `sample` phase, so it shares the policy rollouts' weights version.

```jsonc
{
  "model": {"variants": {"default": {"type": "lora", "rank": 64}, "ema": {"type": "ema", "of": "default", "decay": 0.999}}},
  "imports": ["my_plugins.anchored"], // registers the program and its judge
  "rollout": {
    "type": "my_anchored",
    "sampler": {...},
    "judge": {"type": "my_pair_judge", "llm": {...}},
    "weight": 0.5,
    "reward": [{"type": "hpsv2", "weight": 0.5}]
  }
}
```

### Validation and Inference

```jsonc
// trainer: default rollout.reward, which must then be non-empty; false disables
"validation_scorer": [{"type": "rational_rewards_t2i", "llm": {...}}],

// inference
"reward": {"type": "pickscore"}
```

A program whose signal is a judge validates with any scorer; contrib may wrap a judge and a fixed anchor field of the row into a scorer (`bind`).

### Migration

```jsonc
// before
"rollout_sampler": {...},
"reward": {"type": "pickscore"},
"validation_reward": {"type": "hpsv2"}
// after
"rollout": {"type": "pointwise", "sampler": {...}, "reward": {"type": "pickscore"}},
"validation_scorer": {"type": "hpsv2"}
```

## Python Spec

### Placement

```python
class LocalReward[I, O](BaseModel):  # device worker: sync, batched
    batch_size: int = 32  # items per run_batch(); the worker fills it across groups

    def load(self, device: torch.device) -> None: ...
    def run_batch(self, items: list[I]) -> list[O]: ...  # rows keep their own sizes; outputs moved to CPU by the worker

class RemoteReward[I, O](BaseModel):  # reward loop: one request per item, concurrent
    @property
    def fields(self) -> set[str]: ...  # row keys it reads; every row of an item is snapshotted to CPU on the device worker

    def load(self, device: torch.device) -> None: ...  # clients only, no device use
    async def run_one(self, item: I) -> O: ...

type Reward[I, O] = LocalReward[I, O] | RemoteReward[I, O]

class RewardContext:
    async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]:
        if isinstance(reward, LocalReward):
            return await self._device(reward, items)  # queued; batched per reward on the device worker
        items = await self._snapshot(items, reward.fields)  # an item is a row, or a tuple / list of rows
        return list(await asyncio.gather(*(self._profiled(reward.run_one(item)) for item in items)))
```

An item carries whatever the reward wants besides rows (`(row, ctx.epoch)`, say); `run_batch` / `run_one` never see a context.

### Slots

```python
class BaseScorer(BaseModel):  # scorer_registry; a Reward[Row, Tensor]
    type: str
    name: str | None = None  # label prefix; default type
    weight: float = 1.0
    normalize: Normalize = IdentityNormalize()  # element-wise on raw; a class may default it (pickscore: affine 1/26)

    @property
    def fields(self) -> set[str]: ...  # row keys it reads: the remote snapshot, and the reward-server's /info

    @property
    def labels(self) -> list[str]:  # components; one component logs as <name>
        return ["score"]

    @property
    def weights(self) -> list[float]:  # per component; 0 = logged only
        return [1.0] * len(self.labels)

    async def score(self, ctx: RewardContext, rows: list[Row]) -> Tensor:  # [N, C] raw, CPU
        return torch.stack(await ctx.run(self, rows))

class BaseJudge(BaseModel):  # judge_registry; a Reward[tuple[Row, Row], float]: P(a ≻ b), 0.5 a tie
    type: str

    @property
    def fields(self) -> set[str]: ...

Scorer = Annotated[BaseScorer, RegistryUnion(scorer_registry, "type")]
Judge = Annotated[BaseJudge, RegistryUnion(judge_registry, "type")]
```

Every registered reward is one slot class and one placement (checked at registration). The loader walks the config tree and calls `load` on each.

```python
@scorer_registry.register("pickscore")
class PickScoreScorer(BaseScorer, LocalReward[Row, Tensor]):
    def run_batch(self, rows: list[Row]) -> list[Tensor]: ...  # [C] each

@scorer_registry.register("rational_rewards_t2i")
class RationalRewardsT2IScorer(BaseScorer, RemoteReward[Row, Tensor]):
    llm: LLMClient

    async def run_one(self, row: Row) -> Tensor: ...

# out of tree
@judge_registry.register("my_pair_judge")
class MyPairJudge(BaseJudge, RemoteReward[tuple[Row, Row], float]):
    llm: LLMClient

    async def run_one(self, pair: tuple[Row, Row]) -> float: ...
```

### Scorer Lists

```python
class Scorers(RootModel[list[Scorer]]):  # a single scorer config is shorthand for a one-item list
    @property
    def labels(self) -> list[str]: ...  # one component: "<name>"; several: "<name>/<label>"; a repeated name raises

    @property
    def weights(self) -> list[float]: ...  # scorer.weight * scorer.weights

    async def score(self, ctx: RewardContext, rows: list[Row]) -> Tensor:  # [N, C] raw, CPU
        return torch.cat(await asyncio.gather(*(s.score(ctx, rows) for s in self.root)), dim=-1)

    def normalize(self, raw: Tensor) -> Tensor: ...  # each scorer's normalize on its columns
```

### Rollout Programs

`sample` returns whatever `score` needs; `score` returns the rows the trainer trains on, their `reward_raw` / `reward` (`[C]`, CPU) written.

```python
class BaseRollout[G](BaseModel):  # rollout_registry
    type: str
    sampler: Trajectory  # the policy sampler: ctx.rollouts() samples with it
    reward: Scorers = Scorers([])  # scorers on every policy rollout

    @property
    def labels(self) -> list[str]:  # what the advantage and the logs see
        return self.reward.labels

    @property
    def weights(self) -> list[float]:
        return self.reward.weights

    async def sample(self, ctx: SampleContext) -> G: ...  # every model call of the group
    async def score(self, ctx: RewardContext, group: G) -> list[Row]: ...  # rewards only

Rollout = Annotated[BaseRollout, RegistryUnion(rollout_registry, "type")]

def write(rows: list[Row], raw: Tensor, normalized: Tensor) -> None: ...

@rollout_registry.register("pointwise")
class PointwiseRollout(BaseRollout[list[Row]]):
    type: Literal["pointwise"] = "pointwise"

    async def sample(self, ctx: SampleContext) -> list[Row]:
        return await ctx.rollouts()

    async def score(self, ctx: RewardContext, rows: list[Row]) -> list[Row]:
        raw = await self.reward.score(ctx, rows)
        write(rows, raw, self.reward.normalize(raw))
        return rows

@rollout_registry.register("round_robin")
class RoundRobinRollout(PointwiseRollout):  # Pref-GRPO
    type: Literal["round_robin"] = "round_robin"
    judge: Judge
    swap_orders: bool = True
    opponents: Annotated[int, Field(ge=2, multiple_of=2)] | None = None
    weight: float = 1.0

    @property
    def labels(self) -> list[str]:
        return ["round_robin", *self.reward.labels]

    @property
    def weights(self) -> list[float]:
        return [self.weight, *self.reward.weights]

    async def score(self, ctx: RewardContext, rows: list[Row]) -> list[Row]:
        wins, raw = await asyncio.gather(
            win_rates(ctx, self.judge, rows, swap_orders=self.swap_orders, opponents=self.opponents),
            self.reward.score(ctx, rows),
        )
        write(rows, torch.cat([wins, raw], -1), torch.cat([wins, self.reward.normalize(raw)], -1))
        return rows
```

Pairwise helpers, shared with out-of-tree programs (`flow_control/training/rollouts/pairwise.py`):

```python
def schedule(n: int, opponents: int | None, rng: random.Random) -> list[tuple[int, int]]: ...
    # all pairs, or each row with the opponents/2 next along a random cycle

async def preferences(ctx, judge: BaseJudge, pairs, swap_orders: bool) -> Tensor: ...
    # [P]; with swap_orders asks both orders in one ctx.run() call: s = (s_ab + 1 - s_ba) / 2

async def win_rates(ctx, judge, rows, *, swap_orders, opponents) -> Tensor: ...
    # [K, 1]: (0.5 + wins) / (played + 1); rng = random.Random(ctx.seed("round_robin", ctx.epoch, ctx.prompt_id))
```

Out of tree, the anchored config above:

```python
@rollout_registry.register("my_anchored")
class AnchoredRollout(BaseRollout[tuple[list[Row], Row]]):  # every rollout against one fresh EMA sample of its prompt
    type: Literal["my_anchored"] = "my_anchored"
    judge: Judge
    weight: float = 1.0

    @property
    def labels(self) -> list[str]:
        return ["judge", *self.reward.labels]

    @property
    def weights(self) -> list[float]:
        return [self.weight, *self.reward.weights]

    async def sample(self, ctx: SampleContext) -> tuple[list[Row], Row]:
        anchor = await ctx.start(ctx.seed("anchor", ctx.epoch, ctx.prompt_id), variant="ema")
        rows, _ = await asyncio.gather(ctx.rollouts(), ctx.admit(self.sampler(anchor)))
        await ctx.decode(anchor)
        return rows, anchor

    async def score(self, ctx: RewardContext, group: tuple[list[Row], Row]) -> list[Row]:
        rows, anchor = group
        s, raw = await asyncio.gather(
            preferences(ctx, self.judge, [(row, anchor) for row in rows], swap_orders=True),
            self.reward.score(ctx, rows),
        )
        judged = rearrange(s, "k -> k 1")
        write(rows, torch.cat([judged, raw], -1), torch.cat([judged, self.reward.normalize(raw)], -1))
        return rows
```

Any other way of calling the model goes the same door: `ctx.start` a row under the variant you want (overwrite its latents and σ to continue from a rollout) and `ctx.admit` a sampler node on it.

### Contexts

```python
class RewardContext:  # what score() gets
    prompt: Row  # the store item, CPU
    prompt_id: int  # its store row id
    k: int  # num_rollouts_per_prompt; 1 in validation and inference
    epoch: int

    async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]: ...  # see Placement
    def seed(self, *parts: object) -> int: ...  # derive_seed(trainer seed, ":".join(map(str, parts)))

class SampleContext(RewardContext):  # what sample() gets; its methods raise once sample() has returned
    async def start(self, seed: int, variant: str | None = None) -> Row: ...
        # the prepared prompt row with fresh noise at σ = 1, its generator seeded, and a RootFrame
        # whose variant Model(variant=None) leaves use; None: submit()'s variant (the policy's)
    async def admit[T](self, gen: Calls[T]) -> T: ...  # joins the executor at its next round
    async def decode(self, row: Row) -> list[str]: ...  # processor.decode_output in place (returns the keys it wrote); the row then moves to rollout_storage
    async def rollouts(self, n: int | None = None) -> list[Row]: ...
        # the policy samples: start(seed("rollout", epoch, prompt_id, i)) under the trainer's observers
        # (GRPO trajectory), rollout.sampler admitted, decoded, row["epoch"]; n defaults to k;
        # a later call continues the indices
```

- Write a program as if every reward were remote. A local reward's items wait for the device worker, which serves them between forward rounds whenever the trainer is not in a training step.
- Programs run no CUDA ops and no collectives; every `ctx` method does its device work on the worker. Collectives (image logging, gathers) run on the main thread after `wait()`.
- Rows are shared, not copied. One thread touches a row at a time: the device worker inside an awaited `ctx` call, the program otherwise.
- A group starts when the device worker has room (lazily, as `runs()` today); its prepared prompt row stays on the device until `sample()` returns.
- A program that awaits a remote reward inside `sample()` holds the batch's `sampled()`, and so the next training step, until it returns.

### Runner

The device worker (main thread) and the reward loop (background thread), owned by the trainer for the whole run; inference opens its own.

```python
class Runner:
    def __init__(self, model: ModelAdapter, processor: BaseProcessor, storage: torch.device, seed: int): ...
        # its Executor runs every variant of the model, in one order on every rank; ctx.seed derives from `seed`

    def submit[G](
        self,
        items: Iterable[tuple[int, Row]],  # (prompt_id, store item), pulled lazily
        sample: Callable[[SampleContext], Awaitable[G]],
        score: Callable[[RewardContext, G], Awaitable[list[Row]]],
        *,
        prepare: Callable[[Row], Row],  # store item -> prepared row on the device, when the group starts
        k: int, epoch: int,
        variant: str | None,  # start(variant=None): the policy's
        sampler: Trajectory | None = None,  # ctx.rollouts(): rollout.sampler
        observers: Sequence[tuple[type, Observer]] = (),  # ctx.rollouts(): the trainer's
    ) -> Pending: ...

    def close(self) -> None: ...

class Pending:
    profile: RewardProfile  # one entry per remote request of these groups

    def sampled(self) -> None: ...  # collective: serve device work until every group, on every rank, left sample()
    def wait(self) -> list[list[Row]]: ...  # collective: until every score() returned; submission order
    def as_completed(self) -> Iterator[list[Row]]: ...  # collective; streaming consumers (the inference report)
```

A device-worker round, inside `sampled` / `wait` / `as_completed`:

```python
def _round(self, pending: Pending, until: Literal["sampled", "scored"]) -> bool:
    self._take_requests()  # from the loop: groups to start, rows to start, generators to admit, decodes, local reward items, snapshots
    busy, finished, failed = self._all_reduce(...)  # MAX, MIN over ranks of `until` for `pending`, MAX
    if failed:
        raise ...  # a failed group fails every rank
    if busy:  # forwards add the adapter's own per-variant syncs
        for key, value in self.executor.round():  # one forward per variant; an empty list where idle
            ...  # resolve the generator's future on the loop
    for reward, items in self._local_batches():  # any batch's, per rank, no collective
        ...  # resolve(reward.run_batch(items)), outputs to CPU
    for request in self._row_requests():  # start, decode (then the move to storage), snapshot
        ...
    return finished
# With nothing local and nothing global to forward, a rank waits up to ~20 ms on its queue
# before the next all-reduce.
```

`RewardProfile` / `reduce_reward_profiles` keep their metrics; an entry is stamped on the loop thread when a request is issued, so `produce_span_s` / `overlap_ratio` measure issue times, not decode times.

### Executor and Sampler

```python
class Executor:
    def __init__(self, model: SamplerModel, variants: Sequence[str | None]): ...  # names as given; None: the active variant
    def admit[K, T](self, key: K, gen: Calls[T]) -> None: ...  # joins at the next round, within micro_batch_size calls
    def round(self) -> list[tuple[K, T]]: ...  # collective: the caller makes every rank call it
    # stream / evaluate: unchanged API, loops over admit + round

class RootFrame(Frame):
    variant: str | None = None  # start()'s variant

# Executor groups calls by call.variant or root.variant and passes the name to use_variant as
# given (None: the adapter's active variant; aliases are the adapter's to resolve).
```

### Trainer

```python
class RolloutTrainerBase:
    rollout: Rollout
    rollout_lookahead: NonNegativeInt = 0  # N: batches sampled ahead (n-step-off)
    validation_scorer: Scorers | Literal[False] | None = None  # None: rollout.reward, which must then be non-empty

@dataclass
class PendingRollouts:
    epoch: int
    policy_version: int
    groups: Pending  # was rollouts + rewards + profile
    cursor_state_before: dict[str, int]

def _start_rollouts(self, epoch: int) -> PendingRollouts:
    state = self._cursor.state_dict()
    ids = self._cursor.take(self.num_prompts_per_epoch, epoch=epoch)[self.rank :: self.world_size]
    groups = self.runner.submit(
        ((i, self._store.get(i)) for i in ids),
        self.rollout.sample, self.rollout.score,
        prepare=lambda item: deep_cast_float_dtype(
            self.prepare_row(item, mode="inference", epoch=epoch), self.model.dtype
        ),
        k=self.num_rollouts_per_prompt, epoch=epoch, variant=self._rollout_variant(),
        sampler=self.rollout.sampler, observers=self._rollout_observers(),
    )
    groups.sampled()  # every model call of the batch ran under the current weights; profile/rollout/* as today
    return PendingRollouts(epoch, self._current_epoch, groups, state)

def _finish_rollouts(self, pending: PendingRollouts) -> list[Row]:
    rows = [row for rows in pending.groups.wait() for row in rows]
    ...  # unchanged: profile/reward/*, wait_at_train_s, rollout/policy_lag, profile_dump
    return rows

# epoch e: keep batches e .. e+N sampled, then train on batch e; weights change only in training
while len(self._pending) <= self.rollout_lookahead:
    self._pending.append(self._start_rollouts(self._next_rollout_epoch()))
rows = self._finish_rollouts(self._pending.popleft())
self._train_on_rollouts(rows)
```

- `_compute_advantages` stacks `row["reward"]` / `row["reward_raw"]`, combines with `rollout.weights`, logs `rollout/raw/*`, `rollout/normalized/*`, `group_std` by `rollout.labels`.
- Train-plan hooks (`_build_train_plan`, `_loss_batched`, `_check_rollouts`; GRPO and endpoint trainers) take `list[Row]`: `rollout.row` is the row, `rollout.reward` is `row["reward"]`.
- Each rank takes `num_prompts_per_epoch / world_size` whole prompts.
- `validate_variants(model, step, default)` resolves `validation_scorer` (`False` → none, `None` → `default`); `load_rewards(trainer, device)` loads every reward in the config tree once.
- Forward batches follow reward return order: not bit-reproducible across runs; seeds are.

Validation, per `validation_variants` entry `name`:

```python
async def sample(ctx: SampleContext) -> list[Row]:
    row = await ctx.start(seed)  # seeds and prepare epoch as _validation_runs today
    await ctx.admit(self.validation_sampler(row))
    await ctx.decode(row)
    return [row]

async def score(ctx: RewardContext, rows: list[Row]) -> list[Row]:
    if scorer is not None and not is_padding(rows[0]):
        raw = await scorer.score(ctx, rows)
        write(rows, raw, scorer.normalize(raw))
    return rows  # padding rows included: the main thread's image logging is a collective

groups = self.runner.submit(items, sample, score, prepare=prepare, k=1, epoch=epoch, variant=name).wait()
# main thread: annotate and log images (validation_log_images cap) and, with validation_log_rewards,
# reward metrics; padding rows skipped
```

Inference: the same two functions over the loader's items, `variant=self.variant`; `as_completed()` writes each non-padding row to the report as it finishes.

### Remote

```python
@scorer_registry.register("remote")
class RemoteScorer(BaseScorer, RemoteReward[Row, Tensor]):
    type: Literal["remote"] = "remote"
    url: str
    scorer: str  # the server's name for it; also the default `name`
    # labels, weights, fields: from GET /info at load(); weight and normalize: the client's

# reward-server: hosts local scorers from its own config
#   GET  /info              {scorer: {"labels": [...], "weights": [...], "fields": [...]}}
#   POST /score/{scorer}    rows (torch.save; torch.load(weights_only=True)) -> [N, C] raw;
#                           requests arriving during a batch go into one run_batch()
```

`LLMClient` and `RemoteScorer` share one aiohttp session layer: retries, capacity wait, `max_concurrency`, loop rebinding.

A remote VAE, if needed again, is a `vae_registry` member (`RemoteVAE`) whose server owns the VAE config; `BaseVAE` carries no remote hooks.

### Removed

- `BaseReward` (`_score` / `_async_score` / `_load_model` / `_unload_model` / `score` / `async_score` / `_row_fields` / `component_labels` / `component_weights`), `RewardResult`, `reward_registry` (→ `scorer_registry` × placement, `labels` / `weights`, `fields`)
- `supports_rollout_overlap`, `prepare_row_for_async` (→ placement; the device worker's snapshot)
- `CompositeReward`, `expose_internal_components`, composite-level `normalize` (→ `Scorers`, a plain list)
- the old `PairwiseReward` (a reward whose pairs the framework scheduled), `GatedReward`, `gate_component`, `rejected_raw` (→ the `Judge` slot and rollout programs; gates are a program's business)
- `submit_reward`, `execute_reward`, `submit_pairwise_reward`, `execute_pairwise_reward`, `PendingRewards`, `_CompositePlan`, `_has_pairwise_child`, `_has_overlap_child`, `_score_blocking`, `RewardLoopThread` (→ `Runner`)
- `Rollout` dataclass, `_reward_writer`, `expand_rollouts(whole_prompts=)` (→ rows carry `reward` / `reward_raw`)
- trainer `reward`, `rollout_sampler`, `validation_reward` (→ `rollout.reward`, `rollout.sampler`, `validation_scorer`)
- `RemoteOffloadable`, `RemoteClient`, `endpoint` on rewards and VAEs, `BaseVAE.async_encode` / `async_decode`, `vae-server`, reward-server `/load` (→ the server owns its config)
- The `<type>` label of a multi-component reward's main component (rational rewards: now `<name>/overall`)
