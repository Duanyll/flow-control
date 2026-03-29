# FSDP2 与 DCP：PyTorch 分布式训练的现代实践

> 本文档面向需要在 `flow_control` 项目中修改分布式训练代码的开发者（包括 coding agent），介绍 PyTorch 的 FSDP2（`fully_shard`）和 DCP（Distributed Checkpoint）机制的核心概念与使用模式。

## 1. FSDP2 概述

FSDP2 是 PyTorch 分布式训练的新一代 Fully Sharded Data Parallel 实现，入口为 `torch.distributed.fsdp.fully_shard`（而非旧版 `FullyShardedDataParallel` 包装类）。

### 1.1 基本原理

FSDP2 将模型参数**按维度切分**分布到多张 GPU 上。在前向 / 反向传播时，通过 all-gather 临时收集完整参数，计算完成后再释放。这样每张 GPU 只需常驻 `1/shard_dim` 的参数显存。

### 1.2 DeviceMesh 与 HSDP

FSDP2 通过 `DeviceMesh` 描述 GPU 拓扑。本项目使用 **HSDP（Hybrid Sharded Data Parallel）** 模式，将 GPU 组织为二维 mesh：

```python
mesh = dist.device_mesh.init_device_mesh(
    "cuda",
    mesh_shape=(world_size // shard_dim, shard_dim),
    mesh_dim_names=("replicate", "shard"),
)
```

- **`shard` 维度**：同一 shard group 内的 GPU 共同持有一份完整参数。
- **`replicate` 维度**：不同 shard group 之间是数据并行关系，各自持有参数的完整副本。

当 `shard_dim=1` 时退化为纯数据并行（DDP），当 `shard_dim=world_size` 时退化为纯 FSDP。

### 1.3 `fully_shard` 的使用方式

`fully_shard` 需要**自底向上**地应用到模型的子模块上，最后应用到顶层模块：

```python
# 先 shard 内部的 transformer blocks
for _, module in model.named_modules():
    if type(module).__name__ in fsdp_layers:
        fully_shard(module, mesh=mesh)

# 再 shard 顶层模块
fully_shard(model, mesh=mesh)
```

模型通过 `_no_split_modules` 或 `_repeated_blocks` 属性声明哪些子模块应作为 FSDP 的最小切分单元。

参见：[`hsdp.py:133-147`](../flow_control/training/mixins/hsdp.py)

### 1.4 `fully_shard` 对权重的副作用

**关键陷阱：`fully_shard` 会破坏已有的权重初始化状态。**

这一点与 PyTorch 官方文档的描述不完全一致。如果你先在 CPU/GPU 上加载了预训练权重，再调用 `fully_shard`，权重**会被损坏**（可能是 PyTorch 的 bug）。因此，当 `shard_dim > 1` 时，不能简单地「加载权重 -> shard」，必须使用下文的 seed checkpoint 机制。

当 `shard_dim = 1` 时，不使用 seed checkpoint，直接先加载权重再调用 `fully_shard` 是可行的，可以简化启动推理任务的流程；但训练任务中涉及不同 rank 权重随机初始化一致性问题，仍然必须使用 seed checkpoint。

## 2. DCP（Distributed Checkpoint）概述

DCP（`torch.distributed.checkpoint`）是 PyTorch 的分布式检查点系统，能够在任意并行拓扑下高效地保存和加载模型状态。

### 2.1 与传统 `torch.save/load` 的区别

| | `torch.save/load` | DCP |
|---|---|---|
| 保存格式 | 单个文件 | 目录（含 `.metadata` + 多个分片文件） |
| 分布式支持 | 需要手动 gather 到 rank 0 | 每个 rank 独立读写自己的分片 |
| 拓扑绑定 | 保存时的并行度 = 加载时的并行度 | 可以在不同并行度之间自由切换 |
| 部分加载 | 不支持 | `DefaultLoadPlanner(allow_partial_load=True)` |

### 2.2 核心 API

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict, set_state_dict,
    get_model_state_dict, set_model_state_dict,
    get_optimizer_state_dict, set_optimizer_state_dict,
)
```

- **`get_state_dict` / `set_state_dict`**：处理 FSDP sharded 参数的 FQN（Fully Qualified Name）映射，将分布式 state_dict 转换为标准格式，或反向设置回去。
- **`dcp.save(state_dict, checkpoint_id=path)`**：分布式保存。每个 rank 保存自己的分片。
- **`dcp.load(state_dict, checkpoint_id=path)`**：分布式加载。DCP 根据当前并行拓扑自动拆分 / 合并分片。
- **`no_dist=True`**：单进程模式保存（用于 seed checkpoint 生成）。

### 2.3 `Stateful` 协议

DCP 支持 `Stateful` 协议：如果传入的对象实现了 `state_dict()` 和 `load_state_dict()` 方法，DCP 会自动调用它们。本项目的 `DcpMixin` 利用了这一点：

```python
class DcpMixin(BaseModel):
    def load_dcp_checkpoint(self, checkpoint_path: str):
        state_dict = {"app": self}  # self 实现了 Stateful 协议
        dcp.load(state_dict, checkpoint_id=checkpoint_path, ...)
```

参见：[`dcp.py:14-34`](../flow_control/training/mixins/dcp.py)

### 2.4 `StateDictOptions` 常用选项

```python
StateDictOptions(
    strict=False,              # 允许 state_dict 中有多余或缺失的 key
    ignore_frozen_params=True, # 跳过 requires_grad=False 的参数（训练 checkpoint 用）
)
```

## 3. Seed Checkpoint 机制

本项目将检查点分为两层：**seed checkpoint**（种子检查点）和 **training checkpoint**（训练检查点）。这是整个分布式训练流程中最关键的设计。

### 3.1 为什么需要两层检查点

1. **`fully_shard` 会破坏权重**：已经加载到 CPU/GPU 上的权重在 `fully_shard` 后会损坏。必须先在 meta device 上构建模型骨架、完成 shard，再通过 DCP 加载权重。
2. **DCP 要求特定格式**：DCP 加载需要 checkpoint 以 DCP 目录格式存储，而原始预训练权重通常不是这个格式。
3. **避免重复存储预训练权重**：训练 checkpoint 只保存可学习参数和训练状态（optimizer、scheduler、dataloader 状态等），不重复保存冻结的预训练权重。这在 LoRA 微调场景下尤其重要——训练 checkpoint 只有几十 MB，而预训练权重可能有几十 GB。

### 3.2 Seed Checkpoint 的生成

Seed checkpoint 在**单进程、CPU 上**生成，使用 `flow-control seed` 命令：

```python
# flow_control/scripts/seed.py
model.load_transformer(device=torch.device("cpu"))
model_sd, _ = get_state_dict(model, [], options=StateDictOptions(strict=False))
dcp.save(model_sd, checkpoint_id=seed_dir, no_dist=True)
```

关键点：
- 在 CPU 上加载完整的预训练权重。
- 使用 `no_dist=True` 以单进程模式保存为 DCP 格式。
- 生成的目录包含 `.metadata` 文件和分片数据。

参见：[`seed.py`](../flow_control/scripts/seed.py)

### 3.3 从 Seed Checkpoint 加载（训练启动）

训练启动时的加载流程（`HsdpMixin.load_transformer_from_seed`）：

```
1. torch.device("meta") 上构建模型骨架    ← 零显存
2. 应用 gradient checkpointing（如启用）
3. fully_shard 子模块和顶层模块            ← 注册 FSDP 通信钩子
4. model.to_empty(device=cuda)            ← 在 GPU 上分配未初始化的显存
5. dcp.load(..., allow_partial_load=True)  ← DCP 填充权重，自动处理分片
```

```python
# hsdp.py: load_transformer_from_seed (简化)
with torch.device("meta"):
    model.load_transformer(device=torch.device("meta"))

fully_shard(module, mesh=self.mesh)   # 对每个 FSDP 层
fully_shard(model.transformer, mesh=self.mesh)  # 顶层

model.transformer.to_empty(device=self.device)  # 分配 GPU 显存
model_sd, _ = get_state_dict(model.transformer, [], options=StateDictOptions(strict=False))
dcp.load(model_sd, checkpoint_id=seed_checkpoint_dir,
         planner=DefaultLoadPlanner(allow_partial_load=True))
```

参见：[`hsdp.py:102-170`](../flow_control/training/mixins/hsdp.py)

### 3.4 Training Checkpoint（训练检查点）

训练 checkpoint 只保存**变化的状态**，不包含预训练权重：

```python
# sft.py: state_dict (简化)
opts = StateDictOptions(strict=False, ignore_frozen_params=True)
state = {
    "transformer": get_model_state_dict(self.transformer, options=opts),
    "optimizer": get_optimizer_state_dict(self.transformer, self._optimizer, options=opts),
    "dataloader": self._dataloader.state_dict(),
    "scheduler": self._scheduler.state_dict(),
    "current_step": self._current_step,
}
```

`ignore_frozen_params=True` 确保冻结参数（预训练权重）不会被保存到训练 checkpoint 中。

加载时流程为：先从 seed checkpoint 恢复预训练权重，然后从训练 checkpoint 恢复训练状态：

```python
# sft.py: run (简化)
self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)  # 第一层
self.make_optimizer_and_scheduler()
if self.resume_from_dir is not None:
    self.load_dcp_checkpoint(self.resume_from_dir)  # 第二层
```

参见：[`sft.py:150-188`](../flow_control/training/sft.py)、[`sft.py:261-278`](../flow_control/training/sft.py)

### 3.5 `shard_dim=1` 的退化情况

当 `shard_dim=1`（纯数据并行）时，`fully_shard` 退化为 DDP 行为，不会对权重做跨 GPU 切分。此时可以直接在 GPU 上加载权重，跳过 seed checkpoint 流程（但仍然推荐使用 seed checkpoint 以保持流程一致性）：

```python
if seed_checkpoint_dir is not None:
    # 使用 seed checkpoint 加载（推荐）
    with torch.device("meta"):
        model.load_transformer(device=torch.device("meta"))
else:
    # shard_dim=1 时的 fallback：直接加载到 GPU
    load_device = torch.device("cpu") if self.hsdp_shard_dim > 1 else self.device
    model.load_transformer(device=load_device)
```

## 4. 完整流程总结

```
┌─────────────────────────────────────────────────────────────┐
│  Seed 生成（单进程，CPU）                                   │
│  flow-control seed config.jsonc                             │
│                                                             │
│  1. CPU 加载预训练权重                                      │
│  2. dcp.save(state_dict, no_dist=True) → seed_checkpoint/   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  训练启动（多进程，GPU）                                    │
│  torchrun flow-control train config.jsonc                   │
│                                                             │
│  1. meta device 构建模型骨架                                │
│  2. fully_shard 注册 FSDP 钩子                              │
│  3. to_empty → GPU 分配显存                                 │
│  4. dcp.load(seed_checkpoint/)  ← 第一层：预训练权重        │
│  5. 创建 optimizer / scheduler                              │
│  6. dcp.load(training_checkpoint/)  ← 第二层：训练状态      │
│     （仅包含可学习参数 + optimizer + scheduler + step 等 ） │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  训练过程中保存 checkpoint                                  │
│                                                             │
│  dcp.save(training_state, checkpoint_id=step_XXXXXXX/)      │
│  只保存可学习参数和训练状态，不保存预训练权重               │
└─────────────────────────────────────────────────────────────┘
```

## 5. 相关代码文件

| 文件 | 作用 |
|---|---|
| [`training/mixins/hsdp.py`](../flow_control/training/mixins/hsdp.py) | HSDP DeviceMesh 初始化、`fully_shard` 调用、seed checkpoint 加载/保存 |
| [`training/mixins/dcp.py`](../flow_control/training/mixins/dcp.py) | DCP checkpoint 加载/保存、checkpoint 轮转 |
| [`training/sft.py`](../flow_control/training/sft.py) | SFT Trainer，展示了两层 checkpoint 的完整用法 |
| [`scripts/seed.py`](../flow_control/scripts/seed.py) | Seed checkpoint 生成脚本 |
| [`examples/fsdp/dcp.py`](../examples/fsdp/dcp.py) | LoRA + FSDP2 + DCP 的独立示例 |
