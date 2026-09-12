# flow_control 模块概览

`flow_control` 是一个用于训练和推理 Flow-matching Diffusion Transformer (DiT) 的工具包。以下按子模块介绍其功能和主要接口。

跨模块的数据约定见 [Batch 字典总览](batch-contract.md)：列出输入、缓存、模型调用与输出阶段的字段、形状、坐标空间、生产者和消费者。

---

## adapters — 模型适配器

为不同的 DiT 架构提供统一的前向传播接口。

| 接口 | 说明 |
|------|------|
| `BaseModelAdapter` | 适配器抽象基类，定义 `load_transformer()` 和 list-only `predict_velocity_batched()`；固定形状输入可合并 forward，不兼容输入会安全回退 |
| `Flux1Adapter` / `Flux1FillAdapter` / `Flux1KontextAdapter` | FLUX.1 系列适配器（含拼接、填充、上下文变体） |
| `Flux2Adapter` | FLUX.2 适配器 |
| `QwenImageAdapter` / `QwenImageEditAdapter` / `QwenImageLayeredAdapter` | Qwen 图像生成系列适配器 |
| `LongCatAdapter` / `LongCatEditAdapter` | LongCat 适配器 |
| `ZImageAdapter` | ZImage 适配器 |
| `SD3Adapter` | Stable Diffusion 3.5 适配器 |
| `Krea2Adapter` | Krea 2（Raw / Turbo）适配器 |
| `HiDreamO1Adapter` | HiDream-O1（像素空间统一 Transformer，无 VAE；full / dev）适配器 |
| `parse_model_adapter(conf)` | 工厂函数，根据配置字典创建对应适配器 |

---

## datasets — 数据集加载

提供多种数据源的读取与写入，支持 Pydantic 类型强制转换。

| 接口 | 说明 |
|------|------|
| `parse_dataset(config)` | 工厂函数，根据配置创建数据集 |
| `parse_datasink(config)` | 工厂函数，根据配置创建数据输出端 |
| `LimitedDataset` | 限制数据集长度的包装器 |
| `CoercedDataset` | 对样本施加 Pydantic 类型强制转换的包装器 |

支持的数据源类型：`lmdb`, `plain_directory`, `pickle_directory`, `raw_directory`, `bucket_directory`, `csv`, `jsonl`, `parquet`, `inline`, `lines`；`prism_layers_pro` 随 `flow_control.contrib.efficient_layered` 一起注册（`imports` 引入）

支持的输出端类型：`lmdb`, `pickle_directory`, `raw_directory`, `bucket_directory`

---

## processors — 数据处理

负责将原始数据编码为模型可用的 batch，以及将模型输出解码回图像。包含 VAE、文本编码器、LLM 等组件。

| 接口 | 说明 |
|------|------|
| `BaseProcessor` | 处理器抽象基类，定义 `load_models()`, `encode_batch()`, `decode_batch()` 等异步方法 |
| `parse_processor(conf)` | 工厂函数，根据配置创建处理器 |

**任务类型** (`task`)：`t2i`（文生图）, `tiled_t2i`（保存 tile 布局与逐 tile 条件）, `t2i_control`（可控文生图）, `inpaint`（图像修复）, `efficient_layered`, `qwen_layered`, `tie`

**预设** (`preset`)：`flux1`, `flux2`, `flux2_klein_4b`, `flux2_klein_9b`, `qwen_image`, `qwen_image_edit`, `qwen_image_layered`, `longcat_image`, `longcat_image_edit`, `zimage`, `sd35_medium`, `krea2_raw`, `krea2_turbo`, `hidream_o1_full`, `hidream_o1_dev`

**组件**：

| 组件 | 说明 |
|------|------|
| `VAE` / `Flux1VAE` / `Flux2VAE` / `QwenImageVAE` / `IdentityVAE` | 变分自编码器（`IdentityVAE` 供像素空间模型使用：latent = 缩放后的像素） |
| `Encoder` / `T5TextEncoder` / `ClipTextEncoder` / `Qwen3Encoder` 等 | 文本编码器（`HiDreamO1Encoder` 仅 tokenizer：产出 token ids 而非 embedding） |
| `LLMClient` | LLM 调用客户端（用于 prompt 改写等） |

---

## rewards — 奖励函数

为 RL 训练（GRPO 等）提供奖励打分。

| 接口 | 说明 |
|------|------|
| `BaseReward` | 奖励函数基类，定义 `score()`, `async_score()` 等方法，支持远程卸载 |
| `parse_reward(conf)` | 工厂函数 |
| `execute_reward()` | 执行奖励计算（支持异步批处理） |
| `execute_pairwise_reward()` | 按原始 `__key__` 汇聚乱序完成的 K 个 rollout 后成对评分 |

内置奖励类型：`clip_score`, `pickscore`, `geneval`, `unified_reward`, `composite`（加权组合）, `pairwise`（成对比较）

---

## samplers — 采样器

配置与执行流程见 [采样器设计](sampler-plan-design.md)。`Sampler` 构造纯数据 `SamplingPlan`，所有请求通过同一条 `sample()` 路径执行。executor 在每轮模型求值时同步各请求和分布式 rank；整幅投影、命名分支、tile 前向、guidance 组合和 solver 更新各有明确的执行位置。

| 接口 | 说明 |
|------|------|
| `Sampler` | 持有 solver、sigma grid、`start`、`transforms`、`guidance`、`projectors`；`plan()` 生成当前请求的执行计划，`sample(model, requests, collector=...)` 懒加载请求并按完成顺序产出 run；`make_run()` 创建独立运行状态 |
| `Start` / `SdeWindow` | `start.source` 用条件图像选择器（`inpaint`/`reference[0]`/`clean`/字面键）选初始 latents；设置 `strength` 时按 at-or-below 规则切片，并用切片点 sigma 重加噪。`sde_window` 按请求 RNG 选择窗口，仅保留窗口内的 eta |
| `SampleRun` | `ctx.latents` 是最终结果，`plan` 是实际执行计划；`run()` 生成完整轨迹的叶子调用，`guided_velocity()` 为训练重算单步，每次创建独立 context |
| `Executor` / `ModelCall` | 收集各 run 的分支、tile 调用，固定 variant 顺序，交给 adapter 按 `model.micro_batch_size` 分块；排空 rank 继续 collective |
| `StepCollector` / `StepRecord` | 调用方提供 `(run, step)` 回调，消费每步 latents、velocity 和结果；sampler 不保留历史 |
| `BasePrediction` / `Predictor` | `bind(batch, negative_batch)` 返回独立运行闭包；闭包在同一生成器内调用子节点并完成本算法。`guidance` 配置可嵌套 `model`、`tiled`、`cfg`、`cfg_pp` 及插件；数字仍表示 CFG scale。CFG++ 的 scale 等参数直接放自身字段 |
| `Calls[T]` / `gather()` | 生成器 yield 叶子调用列表，接收同序 fp32 velocity，return 算法结果；`gather` 仅汇合独立子生成器，支持多轮和嵌套，不处理模型、microbatch 或 collective |
| `TiledPrediction` / `MomentumGuidance` | tiling 在自身生成器内切片、求值、拼接；Momentum 包装任意 child，每次求值后更新闭包中的 EMA。CFG 分支、tile、sample 各自 bind，状态自然隔离；同一 Momentum binding 并发求值会报错 |
| `BaseProjector` / `DifferentialDiffusion` | `pre_transition` 每步执行一次；`post_combine` 每次模型求值后执行。Differential diffusion 使用整幅 inpaint mask，控制 `source`（默认 `inpaint`，条件图像选择器）latent 的释放时机 |
| `TiledT2IProcessor` / `TileConfig` | `task="tiled_t2i"` 的顶层字段 `tile_size`/`overlap`（正方形像素，需为 packed stride 倍数），布局用 `utils.tiling.plan_tiles` 在 token 网格上规划；写入 `batch["tiling"]`（序列化的 `TileLayout` 字典）、`batch["model_image_size"]`（单 tile 实际尺寸）与逐 tile 条件 `tiles`；`save_negative=true` 时逐 tile 负条件写入 `negative["tiles"]`（默认关闭） |
| `BasePrediction.velocity()` | 任意预测树的独立 timestep 求值入口；不认识 tiling，也不推断训练模式。SFT 经 `TrainingPredictionMixin.predict_training()`、`EndpointTrainer` 的连续 timestep 直接调用它，都使用显式 `train_predictor` |
| `derive_seed()` | 确定性种子派生 |

`plan.py` 包含 `Transition(solver, sigma, sigma_next, eta)`、求值协议、`StepContext` 和 Euler 原语。执行位置由 `StepContext.item_index/num_items` 提供；solver 的运行历史与逐步公式在 `solver/<name>.py`。`shift` 支持裸数字（`"shift": 3.0` 即 constant shift），默认因子 1.0。分辨率相关 shift 读取 `batch["model_image_size"]`（缺省等于 `image_size`），tiled batch 因而按单 tile 的尺寸/序列长度计算。

**Solver** (`solver.type`)：`flow`（Flow-GRPO SDE/Euler）, `dance`, `ddim`, `cps`, `dpm`（确定性多步 DPM）, `flow_unipc`（UniPC 多步 + UniC 校正）, `sa`（SA-Solver 随机 PEC）, `flash`（逐步重加噪与可选噪声截断）。

GRPO 的 `training/grpo_sampling.py` 提供 `GrpoCollector` 与 `replay_steps(model, items)`：记录 `eta > 0` 的 transition，按实际 eta、Flash ramp 和执行步号重算 `StepLogProbOutput(log_prob, mean, std_dev)`。支持 flow / ddim / cps / dance / flash；rollout 可含 Momentum，但独立训练树必须无状态。分母保留实际 rollout score，current/reference 使用训练树，因此不同树的初始 ratio 不保证为 1。

五个 diffusion trainer 必须显式配置 `train_predictor`：`"model"` 直接前向，`"tiled"` 切片后拼接，CFG 可自由组合。所有 current/old/reference/cache 前向都使用该树；GRPO 与 `EndpointTrainer` 的网格 timestep 用 `make_run(predictor=...)` 保留 rollout 的实际计划及 post-projectors。只有配置含 tiling 时才切片。`model.micro_batch_size` 限制真实前向大小，`train_micro_batch_size` 决定每次 backward 的逻辑 loss 项数。只有 adapter 内允许低精度计算；sampler 与 loss 统一 fp32，低精度存储值在使用前升 fp32。

---

## training — 训练

提供多种训练范式的 Trainer，以及分布式训练、检查点、EMA 等公共功能。

### Trainer

| 接口 | 说明 |
|------|------|
| `SftTrainer` | 监督微调（SFT），支持时间步加权、EMA |
| `GrpoTrainer` | 组相对策略优化（GRPO）：直接继承 `RolloutTrainerBase`，训练点在轨迹上、old 是记录的 log-prob，保留自己的 replay 与 `grpo_loss` |
| `NftTrainer` | Negative-aware Fine-Tuning：`EndpointTrainer` preset，`NftObjective` + `GridTimesteps`（random）+ old-teacher EMA |
| `AwmTrainer` | Advantage Weighted Matching（优势加权的 flow-matching 策略梯度）：`EndpointTrainer` preset，`AwmObjective` + `GridTimesteps(count=6, window=0.9, exclude_first, stratified)` + TRPO-EMA |
| `RamTrainer` | Reinforce Adjoint Matching（KL 正则最优控制的闭式回归目标）：`EndpointTrainer` preset，`RamObjective` + `ContinuousTimesteps(count=8, power_law)` + lagged EMA，不裁剪梯度 |
| `VaeTrainer` | VAE 训练 |
| `Inference` | 批量推理 + 评测（DCP/EMA 权重加载、reward 汇总与逐样本 CSV、datasink/预览输出）。新任务先写 config 走 `launch`，不要另写推理脚本 |

### RL trainer 分层（`rollout_trainer.py` / `endpoint.py` / `train_timesteps.py`）

- `RolloutTrainerBase[ItemT]`：四个 RL trainer 共用的唯一循环——字段块、optimizer/scheduler、验证 EMA、`InitBackupOptimizer`（仅当 `_needs_reference()` 且 `peft_lora_rank == 0`；EndpointTrainer 看 objective 是否需要 `ref`，GRPO 看 `kl_beta > 0`）、`state_dict`/`load_state_dict`、`_train_on_rollouts`、`run()`（含 `validation_non_ema`）。子类实现 `_build_train_plan(rollouts)` 与 `_loss_batched(items, rollouts, advantages)`，可选 `_precompute`。
- `EndpointTrainer(RolloutTrainerBase)`（`endpoint.py`，与 objective 契约同文件）：在 rollout 终点上训练的族，字段 `objective`、`train_timesteps`、`ema_old`。前向规则：`grid_index` 非 `None` 的 item 走 `SampleRun.guided_velocity(x_t, grid_index)`（rollout 的实际 plan + `train_predictor`），连续 timestep 走 `train_predictor.velocity`；old/ref 速度按 `objective.required_policies()` 缓存在 `item.cache`（`precompute_aux_model_outputs`）或 `no_grad` 现算。
- `Objective`（registry union，`"type": nft | ram | awm | weighted_fm`）：纯数学，`compute(TrainPoint, PolicyVelocities) -> LossOutput`，加 `rollout_policy()`（`current`/`old`，谁采样）与 `required_policies()`（`old`/`ref`）。方法超参（`beta`、`kl_beta`、`reward_multiplier`、`off_policy`……）都在这一块。契约（`BaseObjective`/`TrainPoint`/`PolicyVelocities`/`LossOutput`/registry）在 `endpoint.py`，每个成员和它的 preset 同文件（`nft.py`/`ram.py`/`awm.py`；`weighted_fm.py` 只有 objective、没有 preset）。
- `TrainTimesteps`（registry union）：`grid`——按下标在 rollout 网格上取最噪的 `window` 比例，`count`/`fraction`、`exclude_first`、`random`/`stratified`；`continuous`——`count` × `TimestepWeighting`。窗是下标比例，所有 rank 的 item 数只依赖步数。
- preset（`nft.py`/`ram.py`/`awm.py`，各自的 objective 在同一文件）：只设 `training_type`、`objective`、`train_timesteps`、`ema_old`、`clip_grad_norm` 的默认值，`launch.type` 不变。配置中的 `"objective"`/`"train_timesteps"` 块必须带 `"type"`（或写裸字符串取全默认）。

### Mixin

| 接口 | 说明 |
|------|------|
| `CheckpointingMixin` | DCP 分布式检查点 |
| `HsdpMixin` | HSDP 分布式训练支持 |
| `LoggingMixin` | 训练日志 |
| `PreprocessMixin` | 数据预处理 |
| `RolloutMixin` | RL 训练的 rollout 生成 |
| `ValidationMixin` | 验证循环 |
| `LaunchConfig` | 分布式启动配置 |
| `distributed_main()` | 分布式训练入口 |

---

## serving — 模型服务

提供推理服务和 Gradio Web UI。

| 接口 | 说明 |
|------|------|
| `ServeConfig` | 服务配置（host、port、模型、处理器、采样器等） |
| `ServingEngine` | 服务引擎，管理模型加载和生成请求 |
| `create_gradio_app(engine)` | 创建 Gradio Web UI |

---

## utils — 工具函数

| 模块 | 说明 |
|------|------|
| `config` | 配置文件加载（JSON/YAML/TOML）、字典合并、Pydantic 模型更新 |
| `types` | PyTorch 类型标注（`TorchDType`, `TorchDevice`）、`OptimizerConfig`, `SchedulerConfig` |
| `logging` | 日志与终端输出：`get_logger()`, `console`, `warn_once()` |
| `tensor` | 张量操作：`deep_move_to_device()`, `tensor_to_pil()`, `pil_to_tensor()` 等 |
| `hf_model` | HuggingFace 模型加载器 `HfModelLoader` |
| `resize` | 图像缩放：`resize_to_closest_resolution()`, `ResolutionList` |
| `condition_image` | 条件图像选择器 `ConditionImage`/`ConditionImageSpec`：`reference[i]`/`control`/`inpaint`/`clean` 或字面键，同时给出像素键与 latent 键；Start、DifferentialDiffusion、CLIP/IQA reward 共用（契约见 docs/batch-contract.md） |
| `upcasting` | 混合精度：`apply_layerwise_upcasting()`, `cast_trainable_parameters()` |
| `lora` | LoRA 适配器工具 |
| `remote` | 远程模型卸载 `RemoteOffloadable` |
| `pipeline` | 数据处理管线框架：`Pipeline`, `PipelineStage`, `DataSource`, `DataSink` |
| `tiling` | 重叠 tile 布局与融合：`plan_tiles()`（均匀铺排、边缘贴齐）、`extract_tiles()`、`stitch_tiles()`（邻边 Hann ramp、归一化）、`TileLayout`（`batch["tiling"]` 契约）；processors 与 samplers 共用 |

---

## scripts — CLI 入口

通过 `flow-control <command>` 调用。

| 命令 | 说明 |
|------|------|
| `preprocess` | 数据预处理管线 |
| `seed` | 初始化种子检查点 |
| `launch` | 启动分布式训练（SFT / GRPO / NFT / AWM / RAM / VAE / Inference） |
| `vae-server` | 独立 VAE 编码服务 |
| `reward-server` | 独立奖励计算服务 |
| `serve` | Gradio 推理 Web UI |
| `export` | 将 DCP 检查点导出为 HuggingFace 格式 |
| `lora` | 在独立 CPU 进程中转换 DCP/Diffusers LoRA 或将 LoRA 融合进底模 |
| `schema` | 生成各 Trainer / Config 的 JSON Schema |
