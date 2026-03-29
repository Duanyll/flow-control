# flow_control 模块概览

`flow_control` 是一个用于训练和推理 Flow-matching Diffusion Transformer (DiT) 的工具包。以下按子模块介绍其功能和主要接口。

---

## adapters — 模型适配器

为不同的 DiT 架构提供统一的前向传播接口。

| 接口 | 说明 |
|------|------|
| `BaseModelAdapter` | 适配器抽象基类，定义 `load_transformer()`, `predict_velocity()`, `forward()` 等方法 |
| `Flux1Adapter` / `Flux1FillAdapter` / `Flux1KontextAdapter` | FLUX.1 系列适配器（含拼接、填充、上下文变体） |
| `Flux2Adapter` | FLUX.2 适配器 |
| `QwenImageAdapter` / `QwenImageEditAdapter` / `QwenImageLayeredAdapter` | Qwen 图像生成系列适配器 |
| `LongCatAdapter` / `LongCatEditAdapter` | LongCat 适配器 |
| `ZImageAdapter` | ZImage 适配器 |
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

支持的数据源类型：`lmdb`, `plain_directory`, `pickle_directory`, `raw_directory`, `bucket_directory`, `csv`, `jsonl`, `parquet`, `inline`, `lines`, `prism_layers_pro`

支持的输出端类型：`lmdb`, `pickle_directory`, `raw_directory`, `bucket_directory`

---

## processors — 数据处理

负责将原始数据编码为模型可用的 batch，以及将模型输出解码回图像。包含 VAE、文本编码器、LLM 等组件。

| 接口 | 说明 |
|------|------|
| `BaseProcessor` | 处理器抽象基类，定义 `load_models()`, `encode_batch()`, `decode_batch()` 等异步方法 |
| `parse_processor(conf)` | 工厂函数，根据配置创建处理器 |

**任务类型** (`task`)：`t2i`（文生图）, `t2i_control`（可控文生图）, `inpaint`（图像修复）, `efficient_layered`, `qwen_layered`, `tie`

**预设** (`preset`)：`flux1`, `flux2`, `flux2_klein_4b`, `flux2_klein_9b`, `qwen_image`, `qwen_image_edit`, `qwen_image_layered`, `longcat_image`, `longcat_image_edit`, `zimage`

**组件**：

| 组件 | 说明 |
|------|------|
| `VAE` / `Flux1VAE` / `Flux2VAE` / `QwenImageVAE` | 变分自编码器 |
| `Encoder` / `T5TextEncoder` / `ClipTextEncoder` / `Qwen3Encoder` 等 | 文本编码器 |
| `LLMClient` | LLM 调用客户端（用于 prompt 改写等） |

---

## rewards — 奖励函数

为 RL 训练（GRPO 等）提供奖励打分。

| 接口 | 说明 |
|------|------|
| `BaseReward` | 奖励函数基类，定义 `score()`, `async_score()` 等方法，支持远程卸载 |
| `parse_reward(conf)` | 工厂函数 |
| `execute_reward()` | 执行奖励计算（支持异步批处理） |
| `execute_pairwise_reward()` | 成对比较评分 |

内置奖励类型：`clip_score`, `pickscore`, `geneval`, `unified_reward`, `composite`（加权组合）, `pairwise`（成对比较）

---

## samplers — 采样器

实现扩散模型的采样/生成策略。

| 接口 | 说明 |
|------|------|
| `Sampler` | 主采样器（Pydantic 模型），支持 CFG、多种 solver 和 shift 策略 |
| `SampleOutput` | 采样结果，包含最终 latent、轨迹、对数概率 |
| `derive_seed()` | 确定性种子派生 |

---

## training — 训练

提供多种训练范式的 Trainer，以及分布式训练、检查点、EMA 等公共功能。

### Trainer

| 接口 | 说明 |
|------|------|
| `SftTrainer` | 监督微调（SFT），支持时间步加权、EMA |
| `GrpoTrainer` | 组相对策略优化（GRPO），用于 RL 微调 |
| `NftTrainer` | Negative-aware Fine-Tuning |
| `VaeTrainer` | VAE 训练 |
| `Inference` | 推理/生成 pipeline |

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
| `upcasting` | 混合精度：`apply_layerwise_upcasting()`, `cast_trainable_parameters()` |
| `lora` | LoRA 适配器工具 |
| `remote` | 远程模型卸载 `RemoteOffloadable` |
| `pipeline` | 数据处理管线框架：`Pipeline`, `PipelineStage`, `DataSource`, `DataSink` |

---

## scripts — CLI 入口

通过 `flow-control <command>` 调用。

| 命令 | 说明 |
|------|------|
| `preprocess` | 数据预处理管线 |
| `seed` | 初始化种子检查点 |
| `launch` | 启动分布式训练（SFT / GRPO / NFT / VAE / Inference） |
| `vae-server` | 独立 VAE 编码服务 |
| `reward-server` | 独立奖励计算服务 |
| `serve` | Gradio 推理 Web UI |
| `export` | 将 DCP 检查点导出为 HuggingFace 格式 |
| `schema` | 生成各 Trainer / Config 的 JSON Schema |
