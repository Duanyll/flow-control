# Sampler/Solver 重构设计：Plan-as-Data

> 状态：已实施（2026-08-21，分支 `sampler-plan-as-data`，Phase 1–5 全部落地）。
> 本文保留为蓝图与验收标准；与实现的少量偏差见文末「实施偏差」。

## 一句话

把 sampler/solver 从**执行器**改成**编译器**：`Sampler.plan()` 产出纯数据的
`SamplingPlan`（一串 transition 对象），base plan 的规划不碰 model、latent、RNG；
per-sample recipe transform 可以用自己的 generator 选择随机 plan。执行循环上移到
独立的 executor。plan 是 `list`，切片、替换、拼接、改写都是普通数据操作——采样
trick 变成"对 plan 的变换"或"对 velocity 的中间件"，而不是往 sampler 里开洞。

## 动机：现状的洞

现有实现（`flow_control/samplers/`）是照搬各论文参考代码的产物，每个 trick 都在
`Sampler` / `BaseSolver` 上开专属洞：

1. **`BaseSolver.step` 参数爆炸**（`solver/base.py`）：
   `velocity / latents / sigma / sigma_next / prev_sample / eta / state / generator`
   混合了三种生命周期——规划期已知的（sigma、eta）、执行期才有的（latents、
   generator）、replay 才需要的（prev_sample、state）。
2. **flow-grpo-fast SDE window**：`Sampler.trajectory_window_size/range` 两个配置项
   + `_select_trajectory_window` + `_run_sampling_loop` 里的逐步 eta 门控 +
   `train_start/train_end` 上四处 `if return_trajectory` 状态机。
3. **`SASolver` 破坏 step API**（`solver/sa.py`）：PEC 需要在预测点上求值，现行
   "每步预算一个 velocity 喂给 solver"的契约装不下，只好自带 `model_fn` 闭包 +
   `step_callback`，`Sampler.sample` 里 isinstance 特判并放弃跨 request batching。
4. **Callback 穿线**：`step_callback`、`report_progress` 都因为循环锁在 sampler
   内部才存在。
5. **运行时状态挂在配置对象上**：历史上的 `MomentumGuidedSampler`（见 git 历史
   `flow_control/samplers/momentum.py`）把 momentum EMA 存在 pydantic 配置实例的
   `_momentum` 上——在多 request 批量采样下会静默共享状态出错。这是"配置期/执行期
   生命周期混淆"的典型事故。

新需求（SDEdit、DDIM inversion、编辑任务的 RL rollout、pixel-space guidance）如果
继续按开洞方式加，以上问题只会加倍。

## 核心概念：两条正交的改写轴

```
┌────────────────────────────────────────────────────────┐
│ Recipe 层（loop 与 sampler 之间）                        │
│   TextToImage / SDEdit / InversionEdit → list[Phase]    │
├────────────────────────────────────────────────────────┤
│ 轴 1：Plan 变换  f(SamplingPlan) -> SamplingPlan        │
│   SDE window / 切片 / invert / time-travel / restart    │
├────────────────────────────────────────────────────────┤
│ 轴 2：Guidance（velocity 中间件）                        │
│   CFG / cfg-renorm / Momentum / Signed RF / CFG++       │
├────────────────────────────────────────────────────────┤
│ Executor：rendezvous 循环，批量前向 + FSDP 对齐 + 记录    │
└────────────────────────────────────────────────────────┘
```

- **轴 1 改轨迹的形状**：作用在 `list[PlanItem]` 上的纯函数，可打印、可单测
  （不跑 GPU 就能 diff sigma 序列 vs 论文伪码）。
- **轴 2 改每次求值返回什么**：作用在分支 velocity 上的可插拔合成规则，可带
  per-trajectory 状态和辅助模型。
- 两轴在 executor 处交汇但互不侵入。

## 类型设计

对"solver 产出闭包"的原始想法做两处修正：

- **不用字面闭包，用 dataclass + 方法**。闭包不能被 `deep_move_to_device` 搬运、
  不能存 CPU、不能 `dataclasses.replace(tr, eta=0.0)` 改写。规划期对象只持有
  float，不持有 tensor。
- **transition 不接收预算好的 velocity，而是 `yield EvalRequest` 向 executor 索要
  求值**。这让 SA-Solver 回归统一 API，同时保住跨 request batching 和 FSDP 对齐。

```python
# samplers/plan.py
from __future__ import annotations

@dataclass(frozen=True, slots=True)
class Transition:
    """一次 latent 空间转移 sigma -> sigma_next。纯数据，可改写、可切片。"""
    solver: BaseSolver            # 共享的 pydantic 配置引用
    sigma: float
    sigma_next: float
    eta: float = 0.0              # 0 = 确定性步
    record: bool = False          # 执行时是否产出 RecordedStep（替代 return_trajectory）
    save_solver_state: bool = False
    # 是否把该步执行前的 solver state 放进 RecordedStep；由 planner 提前决定

    def run(self, ctx: StepContext) -> Generator[EvalRequest, GuidanceOutput, TransitionResult]:
        return self.solver.run_transition(self, ctx)

# samplers/solver/flash.py —— solver 专属的 transition 子类与其 solver 同住
@dataclass(frozen=True, slots=True)
class FlashTransition(Transition):
    """Flash 仍需一次 model eval；逐步 noise scale 在 planner 中编译。"""
    noise_scale: float = 1.0

@dataclass(frozen=True, slots=True)
class RenoiseTransition:
    """零求值地回到更高 sigma；不可 record，不持有 solver。"""
    sigma: float
    sigma_next: float

    def run(self, ctx: StepContext) -> Generator[EvalRequest, GuidanceOutput, TransitionResult]:
        ...

PlanItem = Transition | RenoiseTransition
SamplingPlan = list[PlanItem]

@dataclass(slots=True)
class EvalRequest:                # transition -> executor："在这个点求一次 velocity"
    latents: Tensor
    sigma: float
    wants_grad: bool = False      # 梯度型 guidance / Hutchinson 散度估计用

@dataclass(slots=True)
class StepContext:                # executor 拥有并维护，per-run
    latents: Tensor
    generator: torch.Generator | None
    solver_state: SolverRuntimeState | None
    guidance_state: GuidanceState | None

@dataclass(slots=True)
class TransitionResult:
    next_latents: Tensor
    recorded: RecordedStep | None = None   # 仅 record=True 且随机步时产出
    next_solver_state: SolverRuntimeState | None = None
    reset_runtime_state: bool = False      # renoise/跳回后清空 solver/guidance 状态
```

### Solver 退化成两件事

**规划**（产出 transition 序列；SA 的 `initial_time`、倒数第二步 sigma 调整在这里
做）和**单步数学**（一个 generator 方法）。Euler SDE 示例：

```python
# solver/flow.py
def run_transition(self, tr: Transition, ctx: StepContext):
    out = yield EvalRequest(ctx.latents, tr.sigma)        # 唯一一次求值
    velocity = out.velocity
    if tr.eta == 0.0:
        return TransitionResult(ctx.latents + (tr.sigma_next - tr.sigma) * velocity)
    mean, std = self._sde_moments(ctx.latents, velocity, tr)
    next_latents = mean + std * randn(..., generator=ctx.generator)
    recorded = None
    if tr.record:
        recorded = RecordedStep(
            latent_t=ctx.latents, latent_next=next_latents,
            log_prob=self._normal_log_prob(next_latents, mean, std),
            replay=FlowReplayStep(
                sigma=tr.sigma, sigma_next=tr.sigma_next, eta=tr.eta),
            solver_state=ctx.solver_state if tr.save_solver_state else None,
            guidance_state=ctx.guidance_state)
    return TransitionResult(next_latents, recorded=recorded)
```

- DPM2/UniPC 等多步法仍有 per-run live state，但统一放在 `StepContext`，不再从
  `Sampler` 到 `BaseSolver.step` 手工穿线。solver/guidance state 都必须按
  functional/immutable 风格更新：新一步构造新 tuple/dataclass，不原地修改传入的
  state 或历史 tensor。这样需要 replay 的 transition 保存旧 state 时只延长 tensor
  生命周期，不必 clone 整份历史。
- live state 和 replay snapshot 是两件事：前者是完成采样所必需的滚动工作集；后者
  只在 `Transition.save_solver_state=True` 的训练步保存。当前支持 replay 的 solver
  都是一阶、无状态，因此全部为 `False`；DPM/UniPC 是确定性多步法，SA 虽随机但
  没有可用的逐步 transition density，三者都不做 logprob replay。字段保留在 plan
  上只是为了将来若出现可计算 density 的随机多步法时无需再改公共接口，当前不为
  假想方法增加实现或测试。
- SA-Solver 的 PEC 在一个 transition 里对预测点二次 `yield`。旧实现首步前多抽的
  无用 randn 不属于算法语义，迁移时直接删除，不为 seed parity 保留 shim。
- `FlashSolver` 每步仍需先求值 velocity/x0，因此使用带 `noise_scale` 的普通
  `FlashTransition`，参与 eta window、record 和 replay；不能降成零求值的
  `RenoiseTransition`。它当前 state 中的 step index 可在规划期编译成每个
  transition 的 noise scale，replay 不需要保存 runtime state。
- `RenoiseTransition` 返回 `reset_runtime_state=True`；inversion 新开 phase 和新的
  `StepContext`。不支持反向运行的 solver 在构造 plan 时直接抛
  `NotImplementedError`，executor 不猜测如何重置或转换历史。

### Executor：rendezvous 保 batching

现行 `_run_sampling_loop` 每步把所有 request 的 latents 合成一次
`get_guided_velocity` 批量前向，其中含 CFG 假前向对齐 FSDP collectives——这段逻辑
必须保留且只存在一份。executor 驱动所有 request 同一步的 generator，收集各自
yield 的 `EvalRequest`，合并成一次批量求值，再 `send` 回去：

```python
# samplers/executor.py
def execute(model, runs: list[Run], guidance: BaseGuidance) -> Iterator[StepEvent]:
    for step_idx in range(num_steps):
        gens = [run.plan[step_idx].run(run.ctx) for run in runs]
        pending, results = prime(gens)  # 同时接住零求值 transition 的返回值
        while pending:
            outputs = eval_and_guide(model, guidance, pending)   # 一次批量前向
            pending = advance(gens, outputs)                     # StopIteration 携带结果
        yield StepEvent(step_idx, results)                       # 替代所有 callback
```

executor 本身是 generator：serving 进度条、trackio 预览、逐步落盘都消费
`StepEvent`，callback 参数消失。分布式校验
（`_validate_distributed_request_count`、negative-pass 同步）迁入，但同步结果是本次
rendezvous 的局部变量，不再写回 `Sampler` 配置实例。

上面是最常见的同构 fast path：所有 run 有相同 transition 数量和 eval 拓扑，sigma
值可以因 resolution shift 不同。合法但少见的异构 plan（不同切片长度、不同 phase）
走 fallback 并 `warn_once`，不报错：rank 内按拓扑分组或顺序执行；跨 rank 的 eval
总数不齐则在每轮 rendezvous 上同步"是否仍有待求值"（同 `_sync_negative_pass` 的
all_reduce 模式），已排空的 rank 用 dummy forward 陪跑到所有 rank 排空。注意
dummy forward 本身只解决 cond/uncond 分支不齐，eval 总数不齐必须靠这层逐轮同步
补齐，否则 FSDP collective 挂死。fast/fallback 的选择也必须跨 rank 一致：dispatch
时对拓扑指纹做一次同步判定，所有 rank 走同一条路径；executor 内部不逐步重复做
capability 检查。

**执行契约**（executor 的硬规则）：

1. 不假设 plan 的 sigma 单调——time-travel/restart 的跳回是合法 plan。
2. 任何 `reset_runtime_state` 的 transition 之后清空 solver/guidance live state
   （多步法历史在 sigma 跳回后是脏的；RePaint 时代没人踩坑只因当年都用一阶法）。
3. Lockstep fast path 的前提是所有 run 共享同一 eval 拓扑；共享 sampler 配置通常
   满足这一点，但 recipe 可以使 plan 异构，此时走上面的显式 fallback（padding
   陪跑 + `warn_once`，不报错）。fast/fallback 的选择跨 rank 一致，由 dispatch 时
   一次同步判定。
4. 一个 request 的全部随机性来自其自身 generator。recipe build 可以用它选择随机
   plan；执行期抽取只发生在 init / transition / guidance 中。新实现内部固定抽取
   顺序，但不把旧实现的 RNG 位序当成接口契约。
5. 零求值 transition（q-sample renoise）合法：generator 不 yield 直接 return。
6. solver state 以 **transition** 为粒度推进：next state 在 transition 完成后一次性
   写回 `StepContext`；执行期间 `ctx.solver_state` 始终是 pre-step state，供
   `RecordedStep` 捕获。
7. guidance state 以 **eval** 为粒度推进：每次 `combine` 返回的 state 立即生效
   （SA 类多次 yield 的 transition，中间 eval 同样推进）。`RecordedStep` 捕获的是
   该 transition **首次 eval 前**的 guidance state——与 replay 单次求值的语义自洽
   （当前支持 replay 的 solver 都是单 eval transition）。

> **已定**（Phase 4b 实现）：规则 2 中 guidance live state 的"清空"实现为用当前
> `execute()` 调用的 guidance 重新 `init_state()`——与 run/phase 开始时的初始化
> 路径一致，stateful guidance 在 sigma 跳回后从干净初始状态继续，而不是以
> `None`（"无状态"）运行；solver state 仍清为 `None`（切片计划从空历史 warmup）。

### 能力边界与检查原则

- plan 构造是 capability 决策的唯一位置：是否允许 inversion、是否支持 step replay、
  哪些 step 要 record/save state，都在执行前写进 plan。
- plan 管线的固定顺序是：solver 产出 full base transitions → 应用 pre-align transform
  （当前仅可选的 `invert`）→ 按 init sigma 对齐 → 应用 window/time-travel 等
  post-align transform → `finalize_replay_state` 收尾。这是对**组装顺序**的约束，不
  限定组装位置：
  简单路径由 `Sampler.plan()` 直接产出 finalized plan；rollout/recipe 需要
  per-sample 变换（如 window 用 per-sample generator 选起点）时，在自己那里变换后
  再 finalize。`finalize_replay_state` 幂等，作为交付 executor 前的固定收尾步无脑
  调用即可；executor 信任收到的 plan，不检查也不重新推导。通用 plan transform
  不重新推导 solver order，也不保存另一套运行时 policy。
- solver 声明的是与配置值无关的静态能力（例如“能否 replay stochastic step”）；
  某个 transition 是否真的随机、是否记录由它自己的 `eta` / `record` 决定。plan 被
  `replace(tr, eta=...)` 改写后，不再回头读取共享的 `solver.eta` 判断能力。
- executor 信任已经构造好的 plan，只保留发现内部编程错误所需的断言，不重复验证
  config，不为合法 config 无法产生的组合增加热路径开销。
- 常见路径使用最简单的 batched 实现；合法但少见的路径允许使用较慢 fallback，并
  给出一次清晰警告。
- 数学上未定义或尚未实现的组合（例如不支持反向的 solver 做 inversion、尚未实现
  replay 的随机高阶方法）在 plan 构造时直接抛 `NotImplementedError`，不扩张公共
  接口去猜测行为。

## Guidance 轴（轴 2）

现行 `Sampler.get_guided_velocity` 揉了两件事：**分支求值**（cond/uncond 批量
前向 + FSDP 对齐，留在 executor）和**合成规则**（可插拔）。拆开后：

```python
class BaseGuidance(BaseModel):          # 注册表模式，同 solver_registry
    def init_state(self) -> GuidanceState | None: ...
    def needs_negative(self) -> bool: ...
    def combine(self, evals: BranchEvals, ctx: StepContext,
                state: GuidanceState | None) -> tuple[GuidanceOutput, GuidanceState | None]: ...

@dataclass(slots=True)
class BranchEvals:
    cond: Tensor
    uncond: Tensor | None
    latents: Tensor
    sigma: float

@dataclass(slots=True)
class GuidanceOutput:
    velocity: Tensor            # 默认 solver 只用这个
    branches: BranchEvals       # CFG++ 型 solver 取原始两支
```

`Sampler.cfg_scale` / `enable_cfg_renorm` 一族退役成 `guidance: Guidance` 联合
类型；旧键不再接受或翻译。省略 guidance 时采用新的
`ClassifierFreeGuidance(scale=1.0)` 默认，不承诺与旧 `cfg_scale=7.5` 默认兼容。

案例谱系（接口即由它们推出）：

| Guidance | 状态 | 额外需求 |
|---|---|---|
| CFG（+renorm） | 无 | — |
| MomentumGuidance（在审） | EMA buffer，per-run | 状态归 `StepContext`，多 request 各自独立——旧实现的 bug 结构性消失 |
| Signed RF（`draft/Signed_RF.pdf`） | 在线 ratio 追踪 uₜ | 同时拿两支 velocity；可持有辅助 classifier（本地小模型，无 collective 问题）；Hutchinson 路线需 `wants_grad` 求值 |
| CFG++ | 无 | solver 从 `branches` 取 uncond 做 renoise——两轴交汇处 |

Pixel-space 模型（HiDream-O1 等）的 latent 即像素，梯度型 guidance 不穿 VAE，
`wants_grad` 路径预期会实际启用。

## RL 消费

`ReplayRequest` / `compute_logprob_at_step` / `SampleOutput.solver_states` 由
`RecordedStep` 取代：

```python
@dataclass(slots=True)
class RecordedStep:               # SampleOutput.trajectory: list[RecordedStep]
    latent_t: Tensor
    latent_next: Tensor
    log_prob: Tensor              # 行为策略 old_log_prob
    replay: ReplayStep            # 纯 float 字段 dataclass，可搬 CPU、可存
    solver_state: SolverRuntimeState | None
    guidance_state: GuidanceState | None

# samplers/solver/flow.py —— ReplayStep 子类与其 solver 同住
class FlowReplayStep(ReplayStep):
    sigma: float; sigma_next: float; eta: float
    def logprob(self, velocity, latent_t, latent_next,
                solver_state=None) -> StepLogProbOutput: ...
```

训练步：用独立出来的分支求值 + `Guidance.combine`（喂回记录的
`guidance_state`）在 `(latent_t, sigma)` 上带梯度重建 velocity，然后
`rec.replay.logprob(...)`。GRPO 的 `_make_replay_request`（手工从 trajectory 切
latent/sigma/翻 solver_states）整个删除——rollout 存下来的就是按步组织好的
`RecordedStep`。目前支持 logprob 的 solver（flow SDE / DDIM-SDE / CPS / Dance /
flash）都不需要保存 tensor 历史：flow、DDIM、CPS、Dance 本身无状态；flash 的
noise scale 在规划期写入 replay 对象。因此当前所有 `RecordedStep.solver_state` 都是
`None`。`save_solver_state` 只保留为 plan 上的扩展点；若未来真有 transition density
可计算的随机多步 solver，再由它的 planner 决定哪些 step 携带 pre-step state。

`RecordedStep` 只服务真正需要逐步 policy replay 的训练（当前主要是 GRPO）。NFT、
AWM、RAM 仍从 `SampleOutput` 读取 final latent 和 executed sigma grid，不要求 transition
record，也不保存 solver/guidance state；sigma grid 是 plan 的轻量元数据，不应和
trajectory tensor recording 绑定。

关键不变量：replay 必须使用与 rollout 相同的输入、分支合成规则和 pre-step state；
数值结果按合理容差比较，不要求跨 microbatch / kernel 的位级一致。stateful guidance
使 pre-state 必须显式记录——在现行架构下意味着 `ReplayRequest` 再开洞，在新设计里
是 `RecordedStep` 的固定槽位。solver state 取执行该步前的版本，guidance state 取该
transition 首次 eval 前的版本（推进粒度见执行契约第 6、7 条）；禁止把 post-step /
post-eval 的 state 记到当前 step。

### Solver 迁移矩阵

| Solver | live state | step replay | `save_solver_state` | 迁移时特别保留 |
|---|---|---|---|---|
| Flow | 无 | `eta > 0` | `False` | 保留 SDE mean/std 语义 |
| DDIM | 无 | `eta > 0` | `False` | `eta=0` 不再抽取随后乘零的无用 randn |
| CPS | 无 | `eta > 0` | `False` | 当前 logprob 是未归一化的 squared-residual objective，原样迁移，不套通用 Gaussian helper |
| Dance | 无 | `eta > 0` | `False` | 一次 randn + Gaussian logprob；`eta=0` 退化为 Euler |
| DPM | x0/sigma 历史 + step metadata | 不支持 | `False` | functional state；plan 切片后从空历史 warmup |
| Flow-UniPC | x0/sigma/last-sample 历史 | 不支持 | `False` | functional state；保留 corrector 的 pre-history 语义 |
| SA | model/time 历史 | 不支持 | `False` | transition 内多次 yield；随机 PEC 没有简单的逐步 Gaussian density |
| Flash | 可编译的 step index | `eta > 0` | `False` | 将实际 noise scale 写入 transition/replay；保留 clipped-noise 近似 logprob |

CPS 和 Dance 都是一阶、无状态 solver，不需要新增 replay snapshot 处理。两者仅在
`eta > 0` 时各抽一次随机数，window 外 `eta=0` 时走确定性 Euler；inversion 若支持
它们，只复用强制 `eta=0` 的一阶确定性路径。CPS 的 logprob 语义与其他 Gaussian
solver 不同，迁移时不能顺手“修正”。Dance 的正 `eta` 公式只对下降 sigma
有定义，非下降 transition 在 plan 构造时直接 `NotImplementedError`；现有 window
排除最后一步的语义也要保留，否则 Dance 会在 `sigma_next=0` 时继续加噪。

## Recipe 层：组合而非枚举

rollout / validation / inference / serving 共享的"造 noise → 调 sampler"构造，
抽象成 `Recipe`。设计原则：**配置层直接镜像底层代数**——底层是"phase 顺序拼接 ×
plan 变换函数复合"，配置层就声明同样的结构，而不是枚举 `TextToImage` /
`SDEdit` 这类写死的变体类。core 只提供组合子；具体 trick 是 jsonc 拼写，不是类。

运行时产物是已经绑定执行依赖的 phase：

```python
@dataclass(slots=True)
class Phase:
    init: InitOp                   # 运行时求值:latents 从哪来
    plan: SamplingPlan             # 已 finalize
    batch: Batch                   # PhaseConfig.batch 已解析
    negative_batch: Batch | None   # 仅该 phase 的 guidance 需要时提供
    guidance: BaseGuidance         # effective sampler 的 guidance
    generator: torch.Generator | None
    # 同一 request 的所有 phase 共享同一个 mutable generator 引用

@dataclass(slots=True)
class RecipeBuildContext:
    default_sampler: Sampler
    batches: Mapping[str, Batch]
    negative_batch_for: Callable[[str], Batch | None]
    generator: torch.Generator | None
```

recipe runner 依次执行每个 `phase.init(..., generator=phase.generator)`，再用同一对象
创建该 phase 的 `StepContext`。所有 phase 持有 `ctx.generator` 的同一个 mutable
引用，因此 build 时的随机 plan、init/renoise 和 solver noise 串在一条 per-request
RNG 流上；generator 不写回任何 pydantic config。

配置侧是三个组合子 registry + 一个通用 recipe：

```python
# InitOp:latents 的来源代数
#   pure_noise | renoise(strength, source) | from_latents(source) | from_previous
# PlanTransform:f(plan, ctx) -> plan 的配置化包装
#   sde_window(size?, range?, record=False) | invert | ...（contrib 随意扩）

class PhaseConfig(BaseModel):
    init: InitOp = Field(default_factory=PureNoise)
    transforms: list[PlanTransform] = []
    batch: str = "main"           # 编辑类 phase 选择另一组条件（消费方备好对应 key）
    sampler: Sampler | None = None
    # None = 继承消费方级 sampler 并**切片**其 plan（同一条轨迹中途进入，标准 SDEdit
    # 语义）；显式给出 = 该 phase 在 [init 起点 sigma, 终点] 上用自己的
    # steps/shift/solver/guidance 造**新网格**（inversion 相位用更多步、编辑相位换
    # guidance 都是纯 config）。

@recipe_registry.register("phases")
class PhasesRecipe(BaseRecipe):   # core 唯一内置成员
    type: Literal["phases"] = "phases"
    phases: list[PhaseConfig] = Field(default_factory=lambda: [PhaseConfig()])

    def build(self, ctx: RecipeBuildContext) -> list[Phase]:
        # 每个 phase:解析 sampler/batch/guidance → full plan → leading invert →
        # 按 init 的起点 sigma 自动对齐 → 其余 transforms →
        # finalize_replay_state；Phase.generator = ctx.generator。
        ...
```

关键规则和边界：

- **起点对齐是 builder 的职责**：phase 的 plan 自动从 init 交付的 sigma 开始
  （SDEdit 的 `strength` 只写一次，不存在 init 与切片点双写失配）。继承 sampler
  时对齐 = 在已有 actual-sigma grid 上切片。没有 leading `invert` 的 phase 自带
  sampler 且要造 partial 新网格时，builder 先用该 sampler 的
  `shift.inverse_sigma()` 把 actual sigma 还原成 canonical `t_start`，再造网格并
  forward shift；不能求逆的 `custom_sigmas` / `diffusers_flow` partial-start 组合
  第一版直接 `NotImplementedError`，不猜边界。
- **shift 的逆是 planner API**：当前 pointwise shift 的主体
  `y = a*x / (1 + (a - 1)*x)` 单调且有解析逆
  `x = y / (a - (a - 1)*y)`；`shift_terminal` 的仿射后处理按相反顺序求逆。
  `BaseShift.inverse_sigma(value, batch, num_steps, t_end)` 与 `apply` 使用同一组
  resolution 参数。只有显式 sampler 从 partial sigma 造新网格时需要它，常见的
  full-grid 和继承后切片路径不增加额外工作。
- **`invert` 在对齐前执行**：下降 full plan 若先按 clean latent 的 `sigma=0` 切片会
  变成空 plan，因此可选的 leading `invert` 先反转 full plan，再由 builder 从 init
  sigma 对齐；SDE/window/time-travel 等其余 transform 在对齐后依配置顺序执行。
  第一版只允许至多一个且必须写在 transforms 首位的 `invert`，其他位置或多次 invert
  在 build 时直接 `NotImplementedError`。这覆盖从 clean latent 做 DDIM inversion
  的目标路径；不落在反转后 grid 上的 partial inversion 起点第一版同样
  `NotImplementedError`，不为更奇怪的组合增加 staging 系统。
- **跨 phase 换 sampler 的数学依据**（rectifiedflow.github.io 博客系列）：所有
  solver 的 state 都钉在直线 RF 的 canonical 坐标 `(x, sigma)` 上——affine 插值
  在时间+尺度重参数化下同流（"All Flows are One Flow"），其离散化版本说明
  DDIM 等价于 warped 网格上的直线 Euler（"DDIM is Straight RF"），ODE 与 SDE 步
  共享边缘分布（"Langevin is a Guardrail"，这条同时是 SDE window 混合
  eta=0/eta>0 步的依据）。重参数化在本库被吸收进 sigma 网格与 solver 内部换算，
  因此 phase 边界递交 `(x, sigma)` 是完备接口：换 solver/shift/步数/随机性只改
  离散化与采样路径，不产生坐标错配。**移植新 solver 的检查项**：它的重参数化
  必须进 planner（网格）或 solver 内部换算，绝不进 state 坐标；多步历史不跨
  solver 边界（phase 重置已保证）。
- **PlanTransform.apply(plan, ctx) 收到 `BuildContext(batch, generator, sampler)`**：
  `batch` / `sampler` 均为该 phase 的 effective 值；window 的随机起点从 per-sample
  generator 抽取。recipe build 的 RNG 位序是新实现自己的稳定语义，不为旧实现的
  seed parity 调整生命周期。轴 1 的纯函数（`with_sde_window` 等）是实现体，
  transform 配置类只是它们的参数载体。
- **反 ansible 边界**：除 leading `invert` 的明确 staging 外，其余 transforms 按列表
  序应用；config 里没有条件、变量、引用。需要逻辑的 trick 把逻辑写进注册的 Python
  （contrib），组合写在 config。
- **随机窗口与 recording 正交**：`sde_window`（省略 size = 全窗）总是负责逐步
  `eta` 门控，但默认 `record=false`；只有 `record=true` 才把同一窗口标为 replay
  轨迹。需要 likelihood replay 的 trainer（当前仅 GRPO）在 build 后校验 plan 至少
  有一个可记录的随机步；NFT/AWM/RAM 可以使用相同随机窗口而不保存 trajectory。

各 trick 的 jsonc 拼写（用了下节的简写规则；dict 全写永远有效）：

```jsonc
// t2i（默认，等价现状）
"recipe": "phases"

// flow-grpo-fast 窗口：是 recipe 里的一个 transform，不是 trainer 字段
"recipe": [ { "transforms": [
  { "type": "sde_window", "size": 3, "record": true }
] } ]

// SDEdit 式 RL rollout（编辑任务）
"recipe": [
  { "init": { "type": "renoise", "strength": 0.6, "source": "clean_latents" },
    "transforms": [ { "type": "sde_window", "record": true } ] } ]

// DDIM inversion 编辑：两个 phase
"recipe": [
  { "init": { "type": "from_latents", "source": "clean_latents" },
    "transforms": [ "invert" ] },
  { "init": "from_previous", "batch": "edit" } ]

// RePaint 式 time travel：contrib 注册 transform 后纯 config 表达
"imports": [ "my_lab.time_travel" ],
"recipe": [ { "transforms": [ { "type": "time_travel", "every": 4, "back": 2 } ] } ]
```

contrib 的替换口有三个粒度：新 `InitOp`、新 `PlanTransform`（覆盖绝大多数
trick）、以及整个 `Recipe`（`recipe_registry` 仍是开放 union，phase 结构本身
装不下的流程注册自定义 recipe，在 `build()` 里做任意 plan 手术）。

## Plan 变换库（轴 1）

```python
def with_sde_window(
    plan: SamplingPlan, start: int, end: int, *, record: bool = False
) -> SamplingPlan:
    result: SamplingPlan = []
    denoise_index = 0
    for item in plan:
        if isinstance(item, RenoiseTransition):
            result.append(item)  # control transition 永远不 record，也不占 window index
            continue
        active = start <= denoise_index < end
        if (
            record and active and item.eta > 0
            and not item.solver.supports_step_log_prob
        ):
            raise NotImplementedError(
                f"{item.solver.type} has no step log-prob replay"
            )
        result.append(replace(
            item,
            eta=item.eta if active else 0.0,
            record=item.record or (record and active),
        ))
        denoise_index += 1
    return result

def finalize_replay_state(plan: SamplingPlan) -> SamplingPlan:
    result: SamplingPlan = []
    for plan_item_index, item in enumerate(plan):
        if isinstance(item, RenoiseTransition):
            result.append(item)
            continue
        result.append(replace(
            item,
            save_solver_state=(
                item.record
                and item.solver.requires_replay_state(plan, plan_item_index)
            ),
        ))
    return result
```

base solver 造 plan 时沿用现有公共语义：最后一个 denoise transition 固定
`eta=0`，其余 transition 才继承 `solver.eta`。因此不加 transform 的 inference、
validation 与普通 rollout 都不会在 terminal step 意外加噪。`sde_window` 在此基础上
进一步门控 eta；`record=true` 才同时替代旧 `return_trajectory` 的记录范围。

窗口随机起点从 `BuildContext` 里的 per-sample generator 抽取，不进 sampler；窗口
index 只数 denoise transition，跳过 time-travel 插入的 Renoise。`ReplayStep` 保存
改写后的实际 `eta`，不能在 replay 时重新读取 `solver.eta`；CPS/Dance 等 solver 的
`eta` 同时影响 mean 和 noise scale。

`with_sde_window` 决定随机范围，并可按显式 `record=true` 标记同一范围；
`finalize_replay_state` 是 plan 管线的收尾步（无论管线在哪里组装），幂等，此时
solver 能看到完整 plan 和 PlanItem index（含 control transition）。任何变换之后、
交付 executor 之前必须（再）调用它；executor 信任收到的 plan，不调用也不检查
这两个函数。`record=true` 遇到 active、随机但不支持 replay 的 solver（当前为 SA）
在 `with_sde_window` 中一次性 `NotImplementedError`，不产生自相矛盾的 plan，也不把
能力检查拖到 executor。

Time-travel（RePaint/FreeDoM 式"每去噪 M 步跳回 N 步"）同样是插入
`RenoiseTransition`（零求值）+ 重放切片的纯函数——它是执行契约第 1、2、5 条的
压力测试，当前不实现，但契约为其预留。control transition 不参与 SDE/record
window；无法定义的 transform 组合在 plan build 时直接 `NotImplementedError`，
不把检查带进 executor。

## 配置面：pydantic 与 registry

分界规则一句话：**出现在 jsonc/schema 里的是 pydantic BaseModel（Sampler /
Solver / Guidance / Recipe / Shift），只活在进程内的是 dataclass（plan / ctx /
recorded / replay）**。config → runtime 的通道只有 `plan()` / `init_state()` /
`build()` 这几个单向构造方法；运行时状态永不写回 config 实例（`_negative_pass` 与旧
momentum 的教训）。serving 在请求边界改 `sampler.steps` / `guidance.scale` 属于
配置变更，合法；反方向禁止。`Transition.solver` 持有 pydantic 配置引用，但
Transition 本身不进 schema，配置面无感知。

### Sampler 前后对比

```python
class Sampler(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = 50
    solver: Solver = Field(default_factory=FlowSolver)
    shift: Shift = Field(default_factory=ConstantShift)   # 因子 1.0 = 不 shift
    sigma_schedule: Literal["linear", "diffusers_flow"] = "linear"
    num_train_timesteps: int = 1000
    custom_sigmas: list[float] | None = None
    seed: int = 42
    guidance: Guidance = Field(default_factory=ClassifierFreeGuidance)   # 新

    # 退役字段：cfg_scale / enable_cfg_renorm / cfg_renorm_eps / cfg_renorm_min
    #          （→ guidance）、trajectory_window_size / range（→ 变换 + trainer 配置）

    def make_sigmas(
        self, batch: Batch, t_start: float = 1.0, t_end: float = 0.0
    ) -> list[float]: ...  # t_start/t_end 是 canonical time
    def plan(self, batch: Batch) -> SamplingPlan:
        return self.solver.plan(self.make_sigmas(batch))
    def plan_from_sigma(self, batch: Batch, sigma_start: float) -> SamplingPlan:
        # linear schedule:actual sigma -> inverse shift -> canonical partial grid
        # custom/diffusers_flow 第一版直接 NotImplementedError
        ...
```

消费方散落的 `sampler.cfg_scale > 1.0` 判断（rollout / serving / inference 三处，
决定是否构造 negative_batch）统一改为 `sampler.guidance.needs_negative()`——判断
权归 guidance，不再裸比较标量。

### 新增 registry：guidance 与组合层三件套

机制照搬 `utils/registry.py` 的 `Registry` + `RegistryUnion` + `Literal` tag，与
shift / solver 同款。新增四个：

```python
# samplers/guidance.py
guidance_registry: Registry[BaseGuidance] = Registry("guidance", base=BaseGuidance)

@guidance_registry.register("cfg")
class ClassifierFreeGuidance(BaseGuidance):
    type: Literal["cfg"] = "cfg"
    scale: float = 1.0
    renorm: bool = False
    renorm_eps: float = 1e-8
    renorm_min: float = 0.0

Guidance = Annotated[BaseGuidance, RegistryUnion(guidance_registry, "type")]

# samplers/recipe.py — 组合层三件套（成员见"Recipe 层"节）
init_op_registry: Registry[BaseInitOp] = Registry("init_op", base=BaseInitOp)
plan_transform_registry: Registry[BasePlanTransform] = Registry(
    "plan_transform", base=BasePlanTransform
)
recipe_registry: Registry[BaseRecipe] = Registry("recipe", base=BaseRecipe)
# core 内置成员：init_op 4 个、plan_transform 若干、recipe 仅 "phases" 一个。
```

`RegistryUnion` 的懒验证意味着在审的组件（MomentumGuidance、Signed RF、新
transform/recipe）可以作为 out-of-tree plugin 经配置 `imports` + `load_plugins`
注册，论文脱敏期不进 core，schema 在 plugin import 后照常生成。

guidance 侧的同款"组合而非枚举"留作后续可选项：把 momentum 这类作用在合成后
velocity 上的东西从"CFG 子类"改成 `middleware: list[...]` 栈（combiner 与
middleware 两个角色本就不同，CFG++/Signed RF 是 combiner，momentum/renorm 是
middleware）。不阻塞本次重构，先用子类。

### 简写规则（RegistryUnion coercion）

在使用**单字段字符串 discriminator** 的 `RegistryUnion` 上统一加三条 coercion，
solver / shift / guidance / recipe / init_op / plan_transform / reward / normalize 等自动
受益。callable/composite discriminator 默认不接受裸字符串，除非该 union 显式提供
coercer。三条简写都无歧义（str / list / number 不可能是合法的成员 dict），纯增量，
dict 全写形式永远有效：

1. **裸字符串 = tag + 全默认**：`"solver": "flow"` ≡ `{ "type": "flow" }`；
   `"init": "pure_noise"`、`"transforms": [ "invert" ]` 同理。
2. **裸 list = 指定的容器成员**：按 union 声明
   `RegistryUnion(..., list_as=(tag, field))`——
   `recipe` 声明 `("phases", "phases")`，`reward` 声明 `("composite", "rewards")`：

   ```jsonc
   "rollout_recipe": [ { "transforms": [
     { "type": "sde_window", "size": 3, "record": true }
   ] } ]
   "reward": [ { "type": "pickscore", "weight": 0.5 },
               { "type": "ocr", "weight": 0.5 } ]
   ```

3. **裸数字 = 指定成员的标量旋钮**：按 union 声明
   `RegistryUnion(..., number_as=(tag, field))`——`guidance` 声明
   `("cfg", "scale")`，`shift` 声明 `("constant", "shift_value")`：

   ```jsonc
   "guidance": 4.5   // ≡ { "type": "cfg", "scale": 4.5 }
   "shift": 3.0      // ≡ { "type": "constant", "shift_value": 3.0 }
   ```

   `bool` 是 `int` 子类，但裸 `true` 一律报错，不当作旋钮值。

不做的糖：无 `type` 的 dict 猜测成员（有歧义）、单元素 list 自动解包（有歧义）。
实现在 `utils/registry.py` 的 `_validate` 入口处理 str/list/number 分支；schema 侧
`__get_pydantic_core_schema__` 生成 `anyOf[union, tag 枚举串, 容器数组, 数字]`，IDE
补全照常。reward 的 list→composite 不依赖 sampler 重构，可先行合入。
`PhaseConfig` 本身不是 union（无 `type` 字段），phase 条目天然无 tag 噪音。

### BaseSolver 配置面的变化

字段面不变（`type` + `eta` + 各自数学参数）；方法面 `step` / `replay_step` /
`init_state` 退役，换成：

```python
class BaseSolver(BaseModel, ABC):
    type: Literal["base"] = "base"
    eta: float = 0.0

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        return [
            Transition(
                solver=self, sigma=sigma, sigma_next=sigma_next,
                eta=self.eta if i < len(sigmas) - 2 else 0.0,
            )
            for i, (sigma, sigma_next) in enumerate(
                zip(sigmas[:-1], sigmas[1:], strict=True)
            )
        ]
    @abstractmethod
    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen: ...
    def requires_replay_state(self, plan: SamplingPlan, index: int) -> bool:
        return False                                                # finalize 钩子
    def invert(self, plan: SamplingPlan) -> SamplingPlan:
        raise NotImplementedError                                   # 能力即方法
    @property
    def supports_step_log_prob(self) -> bool: ...                   # 类级静态能力，保留
```

上面的默认 `plan()` 只供 `eta` 直接表示逐步随机强度的 Flow / DDIM / CPS / Dance
复用，并保持 terminal `eta=0`。其他 solver 的 planner 必须把**实际逐步语义**写进
transition：DPM/UniPC 始终写 `eta=0`；SA 写目标 sigma 上的实际 `_tau(sigma_next)`
且 terminal 为零；Flash 产出 `FlashTransition`，同时写 eta gate 和按 step 编译的
`noise_scale`。因此 transform 永远只读 `Transition.eta`，不根据 solver 类型或共享
配置重新推导某一步是否随机。

`supports_step_log_prob` 保留为"与配置值无关的静态能力"（能力边界节），由
`with_sde_window` 等变换在 plan 构造时消费；`SolverState` 类型从所有公共签名中
消失（runtime state 是 dataclass，只经 `StepContext` 流动）。

### Recipe 与 Sampler 的归属

概念上 sampler 是 **phase 的属性**（一条轨迹可以分段用不同的离散化与
guidance）；工程上采用**继承 + 覆写**：消费方级 `sampler` 字段是所有 phase 的
默认值，`PhaseConfig.sampler` 按需覆写。简单场景的配置面与现状完全一致，
inversion 相位加步数、编辑相位换 guidance 则是纯 config：

```python
class RolloutMixin(...):
    rollout_sampler: Sampler
    # 普通 rollout 沿用 base plan 的随机性，但不保存 trajectory。
    rollout_recipe: Recipe = Field(default_factory=PhasesRecipe)

class GrpoTrainer(RolloutMixin, ...):
    # GRPO 才显式打开全窗 recording；size/range 可由 config 覆写。
    rollout_recipe: Recipe = Field(
        default_factory=lambda: PhasesRecipe(
            phases=[PhaseConfig(transforms=[SdeWindow(record=True)])]
        )
    )
```

不把 sampler 整个折叠进 recipe（只留一个 `recipe` 字段）的原因：95% 场景多一层
嵌套；serving 在稳定路径上运行时改 `sampler.steps`；同一 sampler 在 rollout /
validation 配不同 recipe 的复用没了。runtime `Phase` 已绑定 effective batch、
negative batch 和 guidance；每个 phase 是一次顺序屏障和一次独立 `execute()`，
其内部仍走 lockstep。`RecipeBuildContext` 只为实际 `guidance.needs_negative()` 的
phase 解析 negative batch，不靠全局 `cfg_scale` 猜测。

window **不是**任何配置类的专属字段——它只是 recipe 里的一个 transform 条目。
（本节曾设计过 `trainer.trajectory_window` 字段，那正是要消灭的"专属洞"从
Sampler 挪到 trainer 的翻版，已废弃。）只有需要 likelihood replay 的 trainer 对
recipe 有额外要求：GRPO 在 build 后校验 plan 至少含一个可记录的随机步，失败时
提示给 `sde_window` 加 `record=true`；其他消费方不做这项检查。

### 配置示例（前后）

```jsonc
// 现状
"rollout_sampler": {
  "steps": 10, "cfg_scale": 4.5, "enable_cfg_renorm": true,
  "trajectory_window_size": 3,
  "solver": { "type": "flow", "eta": 0.7 }
}
// 重构后
"rollout_sampler": {
  "steps": 10,
  "guidance": { "type": "cfg", "scale": 4.5, "renorm": true },
  "solver": { "type": "flow", "eta": 0.7 }
},
"rollout_recipe": [ { "transforms": [
  { "type": "sde_window", "size": 3, "record": true }
] } ]
```

### 旧键删除

`cfg_scale` / `enable_cfg_renorm` / `cfg_renorm_*` 与
`rollout_sampler.trajectory_window_*` 均已从配置面删除，不再翻译或静默忽略
（包括值为 `null` 的旧 window 键）。Pydantic 的 `extra="forbid"` 会直接拒绝它们；
新配置分别使用 `guidance` 子对象和 recipe 的 `sde_window` 变换。改动后运行
`uv run flow-control schema` 重新生成 jsonc schema。

## 迁移路径

### 验证策略：语义优先，seed parity 不作为目标

迁移 harness 用来发现数学或状态时序回归，不反过来约束架构。确定性路径比较逐步
latent / mean；随机路径比较给定输入与噪声下的 mean / std / logprob，并检查 RNG
抽取次数符合新实现自身的契约。能够自然得到位级一致时可以用 `torch.equal`，跨
microbatch / kernel 或需要兼容旧 RNG 位序时只要求数值一致。

不引入 RNG compat shim，不保留乘零或完全未使用的 randn，也不为旧 window randint
时机改变 recipe 生命周期。重构允许成为 seed-breaking change：同 config 同 seed
不保证生成与旧实现相同的图像；需要保留的是 solver 数学、随机分布、record 范围和
replay objective。CPS 尤其保留现有 squared-residual logprob，Dance 保留 Gaussian
logprob；这类既有训练语义不能借重构顺手“修正”。

### 阶段

分阶段，每阶段可独立合入：

1. **内部替换**：`Sampler.sample()` / `compute_logprob_at_step()` 公共 API 不动，
   内部改为 plan + executor。对 flow / ddim / cps / dance / dpm / unipc / sa /
   flash 跑数值语义 harness；随机 solver 核对逐步 mean/std/logprob 与抽样分布，不
   为旧 RNG 流添加兼容代码。
2. **RL 切换**：GRPO 的 step replay 改走 `RecordedStep`；NFT/AWM/RAM 改读新
   `SampleOutput` 的 executed-plan 元数据，但不打开 state recording；随后删除
   `ReplayRequest` / `solver_states`。
3. **Guidance 拆分**：`cfg_scale` → `guidance` 联合类型；旧键直接拒绝，省略
   guidance 时使用新默认 `CFG(scale=1.0)`。
4. **Recipe 层**：落地 SDEdit / inversion；消费方（rollout / validation /
   inference / serving）逐个切换。
5. **清理**：删除 `trajectory_window_*`、`sample(t_start, t_end,
   return_trajectory)` 参数和 SA 特判分支。config 面的删除是 breaking change，
   不保留 deprecated 映射。

## 验收标准

迁移完成的判据：`Sampler` 配置面上**不残留任何单一 trick 的专属字段**。现存
洞的归位清单：

| 现状的洞 | 归位 |
|---|---|
| `trajectory_window_size/range` | `with_sde_window` plan 变换 |
| `custom_sigmas` / `sigma_schedule="diffusers_flow"` | planner（sigma 网格构造） |
| `enable_cfg_renorm` 一族 | `ClassifierFreeGuidance` 的字段 |
| 消费方散落的 `cfg_scale > 1.0` 判断 | `guidance.needs_negative()` |
| `FlashSolver` 全量 re-noise + live step index | 带编译后 `noise_scale` 的 `FlashTransition` |
| SA-Solver 特判 + `model_fn` / `step_callback` | 统一 `run_transition` + `StepEvent` |
| `MomentumGuidedSampler`（已删） | stateful `Guidance`，状态在 `StepContext` |
| solver live state / replay snapshot 混用 | live state 仅在 `StepContext`；`Transition.save_solver_state` 决定是否进入 `RecordedStep` |
| `sample(t_start, t_end)` | plan 切片 / Recipe |
| `return_trajectory` + `solver_states` | `Transition.record` + `RecordedStep` |

所有 `solver_registry` 中公开注册的 solver 都必须出现在迁移矩阵和数值语义 harness；
不能只覆盖当前最常用的 config。

## 实施偏差

实现与本设计的有意偏差（均为实现期决定，语义不变或更严格）：

- `ReplayStep` 基类直接携带 `sigma` / `sigma_next`（设计示例放在 `FlowReplayStep`）。
- `plan.py` 只留 solver 无关的协议：transition/eval/record 类型 + 全 solver 共用的
  `euler_step` / `zero_log_prob` / `normal_log_prob`。solver 专属的 transition 子类、
  runtime state、逐步公式（`XxxSolver.step_parts` 等 `@staticmethod`）和 `ReplayStep`
  子类都住在 `solver/<name>.py`，与其 solver 同一个文件；`ReplayStep` 子类通过类名
  调用（如 `FlowSolver.step_parts(...)`），只是代码依赖，descriptor 本身仍是纯 float。
  eta==0 的确定性 replay 由基类方法 `ReplayStep._deterministic()` 统一提供。
- `BranchEvals` / `GuidanceOutput` 定义在 `plan.py` 而非 `guidance.py`（避免循环导入）。
- 执行契约规则 2 的 guidance「清空」= 用当前 `execute()` 的 guidance 重新
  `init_state()`（见 executor 节「已定」注）。
- DPM 迁移时删除了不可达的 `lower_order_second` 死代码分支。
- 记录含随机步的 SA window（旧 `return_trajectory` + eta>0 会静默产出无 log-prob
  的轨迹）现在在 `with_sde_window` 构造期直接 `NotImplementedError`。
- `supports_step_log_prob` 是类级静态 `ClassVar` 而非 property。
- serving 保留 `cfg_scale` 作为请求级 UI 参数，仅映射到 CFG guidance 的 `scale`；
  非 CFG guidance 时忽略并 warn。
- recipe runner（`run_phases`）的 `SampleOutput.timesteps` 取 plan 编译后的
  执行网格：SA 的头部 `initial_time` 与倒数第二格调整会反映在其中；
  `Sampler.sample` 仍报告原始 config 网格（与旧实现一致）。trainer 侧消费的
  轨迹全部来自 recipe 路径。
