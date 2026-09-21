# Qwen-Image-2.1 接入调研

调研日期：2026-09-21。Qwen21 模型部分仍是接入方案；尚未实现其 adapter、更新依赖或运行模型。通用运行时缓存及 Flux1/Flux2 位置 ID 复用已实现于 `feat/adapter-cache-qwen21`。

结论：适合作为独立的 `qwen21` 架构接入，复用现有训练器、数据缓存与任务接口。
主要工作是新 DiT adapter、多模态 encoder、RGBA VAE/preset，以及一个 sigma 网格扩展。
KV cache 按下文的 row 运行时字段方案接入推理和 rollout；训练反传中的缓存复用暂不设计。

## 上游状态与依赖

- [Diffusers PR #14804](https://github.com/huggingface/diffusers/pull/14804) 已于 2026-09-18 合并，提供 `QwenImage21Pipeline`、`QwenImage21Transformer2DModel` 和 `AutoencoderKLQwenImage21`。
- 本次读取的 Diffusers main 为 `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`。下面的源码链接固定到该提交。
- 本次模型仓库 revision 为 `b3179ad355be050328e483a9dfdd9e60cd62adfa`；实施时应一并固定权重、processor 和配置版本。
- 本仓库 `uv.lock` 和已安装环境均锁定 Diffusers `d1b988b08f51110670a0a187f998470d913fb85a`（2026-07-07）；虽然版本字符串是 `0.39.0.dev0`，其中没有上述三个类。
- 本地 Transformers 为 `5.9.0`；[官方安装说明](https://huggingface.co/Qwen/Qwen-Image-2.1#installation) 要求 `>=5.17`。实施时用 `uv add` 更新版本要求，用 uv 更新 Diffusers Git 锁定提交，并检查其他模型的导入兼容性。

## 与仓库现有 Qwen 的区别

| 部件 | 现有实现 | 2.1 接入要求 |
| --- | --- | --- |
| DiT | `adapters/qwen/base.py` 的旧双流模型 | 独立的 32 层、约 7B 单流模型 |
| 文本/视觉编码 | Qwen2.5-VL；Krea 另有 Qwen3-VL 多层特征编码器 | Qwen3-VL 8B，输出 `[B, L, 4096]`，携带图像槽位掩码 |
| VAE | `AutoencoderKLQwenImage`，16 通道 latent，8 倍空间压缩 | `AutoencoderKLQwenImage21`，64 通道 latent，16 倍空间压缩，RGBA 输入输出 |
| latent packing | `patch_size=2` | `patch_size=1`；只是展开空间维度 |
| 编辑序列 | 目标 latent 在参考图之前 | 参考图在前、目标图在后；参考 latent 插入多模态序列的图像槽位 |
| 默认采样 | 示例为 28 步、CFG 4 | 上游为 40 步、CFG 1；负向分支默认关闭 |

参数依据：[Transformer 配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/transformer/config.json)、[VAE 配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/config.json)、[官方架构说明](https://github.com/QwenLM/Qwen-Image-2.1#architecture)。
两个版本每个图像 token 都覆盖 16×16 像素，最终 latent 都可能呈 `[B, HW/256, 64]`；形状相同不代表编码语义兼容，旧预处理缓存和旧 LoRA 不能直接复用。

## 最小接入设计

以下名称均为建议，尚未注册。

1. **独立 adapter**：新增 `flow_control/adapters/qwen21/`，注册 `qwen21_base`，直接继承 `BaseModelAdapter`。一个 adapter 同时处理 T2I 和可选参考图的 TIE；不要继承旧 Qwen 的 RoPE patch、LoRA 配置与目标图切片逻辑。
2. **encoder**：在 `flow_control/processors/components/encoder.py` 注册专用 `QwenImage21Encoder`。可提供返回 embeddings、padding mask、image-slot mask 的方法；preset 覆盖 `encode_prompt()`，沿用 `Krea2RawPreset` 的扩展模式，保持 `BaseEncoder.encode()` 接口稳定。
3. **VAE**：在 `flow_control/processors/components/vae.py` 注册专用 wrapper，复用/提取旧 Qwen wrapper 的帧维度与 mean/std 归一化逻辑。新配置用 `in_channels`，旧 wrapper 读的是 `input_channels`，不能只替换模型类名。
4. **preset**：新增 `qwen_image21` / `qwen_image21_edit`，同时设置 processor 和 adapter 的 `patch_size=1`、`vae_scale_factor=16`、`latent_channels=64`，尺寸保持 32 的倍数。复用 `t2i` / `tie` 任务；编辑 preset 覆盖参考图 resize，保持每张图自己的宽高比，并让 VLM/VAE 使用同一份 resized image。现有 `match_latent` 会强行匹配目标尺寸，不能照搬。
5. **注册与示例**：补全 `adapters/__init__.py`、`processors/__init__.py`，添加机器无关的 preprocess/inference/SFT 示例，运行 `uv run flow-control schema`。`training/inference.py` 应继续通过通用 sampler 工作。

建议 batch 新增 `prompt_embeds_mask` 和 `image_pad_mask`，保留现有 `reference_latents`、`reference_sizes`。
adapter 构造 `img_shapes=[参考图..., 目标图]`，在 encoder 的图像槽位掩码末尾追加 `N_target/4` 个 True；上游 transformer 将每个图像槽位展开为四个 latent token。
传入 `[参考 latent..., noisy_latents]`，使用 `[0,1]` timestep，最终只取输出末尾 `N_target` 个 token。
参见 [Transformer.forward / build_token_metadata](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/transformers/transformer_qwenimage21.py)。

### 编码与 RGBA 的正确性要求

- 原样匹配上游 raw prompt template、system-prefix 裁剪、左 padding，以及 `mm_token_type_ids` 的传递。
- 需要最后 decoder 层 **最终 RMSNorm 之前**的特征。上游在 Transformers 5.x 使用临时 norm hook；当前 Krea 的 `Qwen3VLEncoder` 取 12 层特征且不支持图像，不能直接复用。
- 透明参考图送入 VLM 前合成到白底 RGB；送入 VAE 的副本保留 alpha。保留多模态 processor 的 patch/merge 配置，不能换成普通 Qwen3-VL processor 默认值。
- 编辑 preset 应设置 `negative_with_images=True`：虽然默认关闭负向分支，用户启用 CFG 时，负向 encoder 也必须读取同一组参考图。否则负向 row 保留了参考 latent，却没有对应图像槽位，与 `img_shapes` 不匹配；官方 pipeline 的负向分支也传入参考图。
- 仓库的 `BaseProcessor._adapt_image_channels()`、PIL/tensor 转换和 Gradio 已有 RGBA 基础支持。仍需验证 PNG、预览与所用 reward 的 alpha 处理；RGB reward 的背景合成策略不能影响保存的 RGBA 原图。
- 官方宣传原生 2K，但本次锁定的 pipeline 源码默认 `output_resolution=1024`。示例应显式指定分辨率；先用 1024 做对齐，再验证 2048。编辑无显式尺寸时，上游按最后一张参考图的宽高比确定输出，通用 TIE 当前使用第一张，需显式约定或覆盖。

编码细节以 [pipeline 的 `_get_qwen_prompt_embeds()` / `__call__()`](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py) 为准；其注释指出将来可用 Transformers 的 `tie_last_hidden_states=False` 替代 hook，实施时应重新核对版本。

### sigma 网格：已确认需要扩展

[官方 scheduler 配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json) 为动态 exponential shift：`base_image_seq_len=256`、`max_image_seq_len=8192`、`base_shift=0.5`、`max_shift=0.9`、`shift_terminal=0.02`。
分辨率 shift 只按目标图 token 数计算，不包括参考图。

官方流程是 `linspace(1, 1/steps, steps)` → shift/stretch → 追加终点 `0`。
本仓库 `Sampler._make_sigmas()` 的 `linear` 对包含末尾零的整个网格调用 shift，因此直接配置 `shift_terminal=0.02` 会令采样停在 0.02。
已有 `diffusers_flow` 则先处理 training-grid endpoints，与该 pipeline 显式传入 sigmas 的方式也不完全相同。

在未改动依赖的环境中，使用已安装 `FlowMatchEulerDiscreteScheduler`、本仓库 `Sampler` 与 40 步网格做了 CPU 数值对照（无模型加载）：

| 分辨率 | 官方末三项 | 当前 `linear` 末三项 | 当前 `diffusers_flow` 最大误差 |
| --- | --- | --- | --- |
| 1024² | 0.06788057, 0.02, 0 | 0.11336720, 0.06782311, 0.02 | 0.00119883 |
| 2048² | 0.10222954, 0.02, 0 | 0.18034768, 0.10527313, 0.02 | 0.00313473 |

按上述官方次序构造候选网格，两种尺寸最大误差均为 `1.19e-7`。另以 AST 对照确认，已安装 scheduler 的 `set_timesteps()`、`time_shift()`、`stretch_shift_to_terminal()` 与本次固定的上游版本一致。
建议新增清晰命名的 sigma schedule 选项，不改变旧配置语义；固定尺寸的首次对齐也可暂用现有 `custom_sigmas`，但它绕过 shift，不适合直接用于可变尺寸示例。
Euler 更新与现有 `FlowSolver(eta=0)` 可复用。

## KV cache 与 FLUX.2 Klein 9B KV 的关系

都是首步提取条件 K/V、后续只计算变化部分。[Klein KV](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_klein_kv_cache.md) 缓存参考图，后续仍计算文本和目标图；Qwen 2.1 缓存文本与参考图组成的整个前缀。

Qwen 的 block-causal attention 保证前缀看不到目标图，图像块内部双向；`causal_condition=true` 又把前缀 modulation 的 timestep 固定为 0。
因此固定权重、条件、布局下，缓存前缀与每步重算应数值等价；纯 T2I 也能缓存文本。上游已有跨 timestep 的 [cache/full-forward 一致性测试](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/tests/models/transformers/test_models_transformer_qwenimage21.py)。

### 推理 / rollout 的缓存约定（基础设施已实现）

**普通 row 字段以 `_` 开头就是运行时缓存**，例如 `_txt_ids`、`_img_ids`，后续 Qwen21 用 `_kv_cache`。这些字段可以直接声明在 TypedDict 中，不需要 holder、slot 或逐字段 getter。
数据标识统一为 `key`，采样补齐标记为 `padding`；`_` 前缀不再承载普通数据。

1. `ModelPrediction.bind()` 为每个 variant 保存一个普通缓存 dict。每次调用新建 `{**row, **cache, "noisy_latents": ...}`，yield 返回后收回其中的 `_` 字段。同一 binding 的并发调用不会覆盖彼此的 noisy latents；CFG 正负分支和 tile 各自 bind，缓存隔离。
2. `BaseModelAdapter._prepared_batches()` 负责设备/dtype 转换与浅回写。只有 `use_cache=True`、Executor 的 ContextVar 开启且无梯度时才带入/回写缓存；缓存对象保持原引用，不递归转换。其他调用正常重算，临时字段在退出时删除。
3. adapter 直接执行 `if "_img_ids" not in batch: batch["_img_ids"] = ...`。Flux1/Flux2 已采用这个写法，并继续支持调用者提供的普通 `txt_ids` / `img_ids`。
4. Executor 仅在实际 forward 期间开启 ContextVar，并在 `finally` 恢复。每轮先 `send()` 让 predictor 收回缓存，再清除旧调用 row 的 `_` 字段；完成/yield、异常与显式 close 同样清理。`SampleRun.run()` 的局部 binding 随轨迹结束释放，输出 row 与 negative row 不携带缓存。
5. dummy 快照在 detach/broadcast 前过滤 `_` 字段，dummy forward 后也删除临时缓存。无需在每个 writer、RPC 或 rollout 输出安装过滤器。

缓存的作用域是一次采样轨迹中的条件分支与 variant。期间权重、条件、设备和 dtype 固定；不做内容 hash、LRU 或权重版本管理。
推理及 no-grad rollout 复用缓存；训练反传复用暂不设计。直接调用 adapter 的公开接口默认不保留缓存。

### microbatch

当前 Flux 的 `_txt_ids` / `_img_ids` 通过 `shared_cache_fields` 声明为布局相同即可共享的字段。通用 dense collator 从兼容样本取已有值，forward 后写回各逻辑样本。
Flux2 保存 `[1, N, 4]`，使用时展开到当前 B，避免交错采样导致 batch 大小变化后用错形状。

**Qwen21 的 K/V 依赖每个样本的条件，不能声明为共享字段。** 下一步先实现 batch=1 的 `_kv_cache`：为空时执行 `extract`，有值时执行 `cached`。它已经能走逐样本回退和上述生命周期。
后续 dense 支持由 Qwen21 adapter 自己组装、拆分缓存，不在通用层预建递归 scatter 框架：

- 除已有 shape 检查外，要求 `image_pad_mask` 内容与完整 `img_shapes`（参考图 H/W、顺序及目标尺寸）一致，因为上游读取第一行布局。
- 全部未 prefill：一次 dense prefill，沿 B 拆分后保存到各 row。
- 全部已有缓存：临时沿 B 拼接每层 K/V，执行 dense decode。
- 混合阶段或布局不同：使用现有跨 rank 一致的逐样本回退。

batch=1 直接保存上游 cache。B>1 prefill 拆分时 clone，避免一条样本保留整个 batch 的存储；dense decode 的临时拼接会增加峰值显存，需要计入基准。

CPU mock 已验证：Flux 缓存/无缓存输出一致、B=2→1、CFG 隔离、variant 切回命中、完成/yield/异常/close 清理，以及 dummy 不传播运行时字段。Qwen21 真模型的逐步数值对齐和缓存性能仍待下一步验证。

BF16、batch=1 下，按配置推算缓存成本为 `32层 × 2(K/V) × 4096 × 2字节`，即每个前缀 token 0.5 MiB。
一张经 resize 后为 1024² 的参考图产生 4096 个 latent token，其常驻 K/V 约 2 GiB，另加文本、激活和合批临时内存；这是张量容量估算，不是实测峰值。多参考图应同时测吞吐与显存。

## 训练与性能验证顺序

1. **依赖/组件**：更新后验证全仓库导入；对齐 encoder 特征与两类 mask、RGB/RGBA VAE encode/decode、latent pack/unpack、官方 sigma 网格。
2. **无缓存推理**：固定初始 latent、prompt、尺寸和 scheduler，逐步比较官方 pipeline 与 adapter 的 velocity/latent；覆盖 T2I、单参考图、多参考图、透明输出。仅相同 seed 不保证随机数布局一致。
3. **训练闭环**：验证 LoRA 梯度、一步 optimizer update、保存/重载与 FSDP。上游有 PEFT、gradient checkpointing 和 `_no_split_modules`，适配仓库机制有基础，但当前尚未做真模型训练验证，也未确认最优微调配方。
4. **缓存/批处理**：验证跨 timestep 等价、不同请求与 CFG/LoRA 分支隔离、异构布局回退、空 rank/dummy forward，以及 rollout/replay 的策略边界。
5. **性能**：基准先保留上游默认分段 SDPA；FlexAttention 作为独立优化。上游特别指出未编译的 flex 路径可能物化巨大的 attention 矩阵，不能只换 processor 就宣称更快。正式 GPU 验证应在 Slurm 作业中运行并监测利用率。

新 LoRA 可先覆盖 `attn.to_q/to_k/to_v/to_out.0`，再评估 `img_mlp.proj/out/gate_layer`；旧 Qwen 的 `add_*`、`txt_mlp` 和 `img_mod` 等目标不适用。
`QwenImage21Rope.freqs` 仍是普通 tensor list，且 forward 中有 host-side 元数据处理；需要检查本仓库 meta/FSDP 加载与 profiler 结果，再决定是否改成 buffer 或缓存位置元数据，不能直接套用旧 RoPE 实现。

Qwen21 部分只做源码、配置和小型 sigma 数值调查；通用缓存与 Flux ID 复用使用 CPU mock 验证。未下载权重，未提交 GPU 作业。吞吐、峰值显存、真实 LoRA/FSDP 兼容性及生成质量仍属于实施阶段验收项。
