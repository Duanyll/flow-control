# Qwen-Image-2.1 与 FLUX.2 Klein KV

更新日期：2026-09-21。已在 `feat/adapter-cache-qwen21` 实现两种模型的 adapter、processor 和推理/rollout KV 复用。

Qwen21 是独立的 `qwen21/base` 架构，FLUX.2 KV 使用 `flux2/kv`；均复用现有训练器、数据缓存与任务接口。KV 是 row 上的 `_kv_cache`，训练反传不复用。

运行示例：

```bash
uv run flow-control launch examples/t2i/qwen21/inference.jsonc
uv run flow-control launch examples/t2i/flux2/klein_9b_kv.inference.jsonc
# 在 GPU 作业内运行官方 T2I、透明图和编辑提示词，另存缓存开关计时：
uv run tests/kv_inference_smoke.py qwen21 --output demo/qwen21
uv run tests/kv_inference_smoke.py flux2_kv --output demo/flux2_kv
```

编辑使用 `task="tie"`，preset 分别为 `qwen21`、`flux2_klein_9b_kv`，输入 `reference_images`；显式提供 `image_size=[H,W]`。不指定时沿用仓库 TIE 的第一张参考图推导规则，与 Qwen 官方按最后一张推导不同。`model.use_cache=false` 可关闭跨步复用。

### 实际样图与编辑问题

2026-09-21 已通过 Slurm 运行两个模型的官方提示词。Qwen 的霓虹招牌 T2I、RGBA 小龙，以及 FLUX 的猫和巫师猫编辑均成功；Qwen 的部分编辑存在上游可复现的质量问题，不能用缓存开关的数值接近代替图像质量检查。

失败样例是把 1024² 霓虹招牌的背景换成夕阳海滩。仓库和锁定版本的官方 `QwenImage21Pipeline` 都输出过度锐化、带光晕的近似原图。官方 pipeline 的独立对照结果：

| 输出尺寸 | 参考图处理尺寸 | 结果 |
| --- | --- | --- |
| 1024² | 1024² / 768² / 512² | 均失败 |
| 768² | 768² | 正常换成海滩 |
| 512² | 512² | 正常换成海滩 |

同为 1024²，透明小龙换海滩、猫戴红围巾则正常。霓虹图在仓库中改用更明确的提示词，或固定 `mu=0.69` 并取消 terminal stretch，仍失败。[Diffusers #14824](https://github.com/huggingface/diffusers/issues/14824) 报告了同版本的相同症状，尚无已确认根因；这不是所有 1024 编辑都会失败的尺寸限制，因此不自动降分辨率或修改默认 sampler。

按用户建议，把同一霓虹样例的输出与参考处理尺寸都设为 **2048²** 后，仓库缓存版和无缓存版均正常换成海滩，视觉接近。48 GB RTX 4090、BF16、40 步、seed 42 下，缓存版去噪 `148.1 s` / 峰值分配显存 `26.51 GiB`，无缓存版 `254.5 s` / `18.50 GiB`，约 **1.72×** 加速；最终 latent 相对 L2 约 `0.00665`。计时不含模型加载、条件编码和 VAE 解码，编码器在去噪前卸载。2K 图片在 `demo/qwen21/2048/`，没有按这个单一样例改变全局默认尺寸。

对齐检查中，prompt embeddings 与 image-slot mask 和官方完全一致。参考图改为官方 PIL RGBA 缩放后，方图、横图、竖图的归一化像素及内存布局都一致；此前 bilinear/连续布局与官方不同，1024 方图的 VAE latent 最大差异 `0.0957` 在相同布局下归零。小龙完整采样与官方视觉接近，最终 latent 相对 L2 为 `0.0114`；仅对齐 timestep 的 BF16 舍入顺序降为 `0.0075`，不是霓虹图失败原因。保留现有 adapter 的 timestep 约定。

原始输出及排查对照保存在 gitignored 的 `demo/qwen21/`、`demo/flux2_kv/`。当前 smoke harness 用透明小龙作为 Qwen 编辑参考图，用猫作为 FLUX 编辑参考图；保留官方原始提示词。

### 专用 prompt enhancer 实验

两个 Qwen3.5-VL 9B 改写模型也已从 ModelScope 下载，并用 `relink_to_hf_cache.py` 硬链接至共享 HF 缓存，每个约 17.5 GiB：

- `Qwen/Qwen-Image-2.1-PE-T2I`，revision `f3ed7985c788ad75b3ab7223e0c4c51e2a43545b`。
- `Qwen/Qwen-Image-2.1-PE-I2I`，revision `72927bc08afc99b7888ceb7d7d51a12db3700bbd`。

本次仅做少量直接 `generate()` 的实验，复用[官方 Transformers runner](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/run_transformers.py)，不启动 vLLM，也未给通用 processor 增加新的服务或配置层。两个模型都通过 `AutoProcessor` / `AutoModelForImageTextToText` 加载，读取各自的 `system_prompt.txt`，开启 thinking；`temperature=1`、`top_p=0.95`、`top_k=20`，T2I presence penalty 为 `1.5`，编辑为 `0`。T2I 模型卡简例遗漏了 presence penalty，因此以官方 runner 的生产 profile 为准。

生成结果保存在 `demo/qwen21/enhancer/{t2i,edit}.jsonl`；必须检查 `parse_ok=true` 后才把 `positive_prompt` 送入 DiT，不能只看 runner 的退出码。`manifest.json` 记录 checkpoint revision、system prompt 与 runner 哈希及采样参数。对比中，T2I 两侧使用相同的推荐画幅；编辑按 1024²、2048² 分别固定参考图处理尺寸，PE 自身缩小后的 RGB 输入不替换 DiT 的原始 RGBA 参考。

两条改写与五张新增配对图片已跑完，结果在 `demo/qwen21/enhancer/comparison.md` / `comparison.png`：

- **T2I 柯基弹吉他**：原始中文短句出现人手扶琴；719 词改写后，柯基本身用带毛前爪弹完整吉他，构图和雨夜氛围更完整。两侧均为推荐的 1696×2528。但改写也主动增加雨衣、咖啡馆、行人和英文招牌，属于创作扩写。
- **1024 霓虹编辑**：153 词改写明确了删除建筑、保留霓虹和倒影，仍没有救回过度锐化的城市近复制。
- **2048 霓虹编辑**：原始与改写均成功换海滩；改写版没有原始版额外的棕榈树，霓虹与倒影更突出。只能说更贴合改写后的约束，单例不足以证明整体画质提升。

图像端统一 seed 42、40 步、CFG 1、BF16、缓存开启。T2I 原始/增强去噪分别约 113/116 秒，2048 增强编辑约 149 秒；没有额外安装生成加速内核。当前仍将 enhancer 保留为实验，未自动改写用户输入。

## 上游状态与依赖

- [Diffusers PR #14804](https://github.com/huggingface/diffusers/pull/14804) 已于 2026-09-18 合并，提供 `QwenImage21Pipeline`、`QwenImage21Transformer2DModel` 和 `AutoencoderKLQwenImage21`。
- 本次读取的 Diffusers main 为 `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`。下面的源码链接固定到该提交。
- 本次样例使用的 Qwen21 缓存 revision 为 `790c92633540aa0cb11d9abf19eb46d861714758`，FLUX.2 KV 为 `a6dfb36eca3a3906eb2fd460795adfb844e5fcce`。
- Diffusers 已固定到上述提交；Transformers 升至 `5.17.0`，safetensors 至 `0.8.0`；Trackio 升至 `0.38.1` 以兼容升级后的 Hugging Face Hub。

权重均从 ModelScope 的同名模型仓库下载（[Qwen21](https://modelscope.cn/models/Qwen/Qwen-Image-2.1)、[Klein KV](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9b-kv)），随后用 `relink_to_hf_cache.py` 硬链接至共享 Hugging Face 缓存，避免重复存储。仅取 Diffusers 的 transformer、encoder、VAE 和配置，未下载 Klein 额外的单文件权重。

Transformers 5.17 的 AutoTokenizer 会先查模型配置，但 Qwen processor 子目录只有 tokenizer 配置；两个 Qwen VLM processor loader 显式传 `tokenizer_type="qwen2"`，支持完全离线加载。

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

## 接入实现

1. `flow_control/adapters/qwen21.py` 注册 `qwen21_base`，同一 adapter 处理 T2I 和 TIE。
2. `QwenImage21Encoder.encode_condition()` 使用只装入 encoder/processor 的轻量官方 pipeline，直接复用 prompt 编码，返回 embeddings、padding mask、image-slot mask。
3. `QwenImage21VAE` 复用 Qwen 的帧维度和 mean/std 归一化，读取新配置的 `in_channels`。
4. `QwenImage21Preset` 注册为 `qwen21`，设置 `patch_size=1`、`vae_scale_factor=16`、`latent_channels=64`，尺寸保持 32 的倍数。参考图按 `total_pixels`（默认 1024²）和官方 `calculate_dimensions` / PIL RGBA 路径保留宽高比缩放，同一张 resized image 交给 VLM 和 VAE；参考面积独立于输出尺寸。
5. `flow_control/adapters/flux2/kv.py` 安装官方 KV attention processors。参考在目标之前；关闭复用时每步走 `extract` 并丢弃 KV，以保留固定参考 timestep 和因果注意力。preset 从 KV 模型仓库加载 encoder/VAE，参考图遵循官方 1024² 面积上限和 16 倍数裁剪。

batch 新增 `prompt_embeds_mask` 和 `image_pad_mask`，保留现有 `reference_latents`、`reference_sizes`。
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

### sigma 网格：修复现有 `shift_terminal`

[官方 scheduler 配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json) 为动态 exponential shift：`base_image_seq_len=256`、`max_image_seq_len=8192`、`base_shift=0.5`、`max_shift=0.9`、`shift_terminal=0.02`。分辨率 shift 只按目标图 token 数计算，不包括参考图。

官方流程是 `linspace(1, 1/steps, steps)` → shift/stretch → 追加终点 `0`。
这里 `shift_terminal=0.02` 指最后一次模型评估的 sigma，最后一步仍从 0.02 更新到 0。

此前本仓库 `BaseShift.apply()` 将包含终点零的整个网格一起拉伸，导致 `linear` 最终停在 0.02。这是现有实现的错误，不需要为 Qwen21 新增 sigma schedule。
现已只拉伸正值 sigma，保留终点零；单步 `[1, 0]` 保持不变，避免把唯一的评估点拉伸而除零。
因此直接用现有 `linear`、上述 `LinearShift` 配置和 `FlowSolver(eta=0)` 即可。

修复前，1024²、40 步的末三个点为 `0.11336720, 0.06782311, 0.02`；官方为 `0.06788057, 0.02, 0`。
CPU 回归检查将普通 `Sampler.make_sigmas()` 与官方 `FlowMatchEulerDiscreteScheduler` 的显式 sigmas 路径比较，覆盖 1024²/2048²、2/40 步，并检查最终 solver transition 到达 0。40 步时两种尺寸最大误差约 `1.19e-7`。

`diffusers_flow` 从训练网格端点构造推理时间表，属于另一种网格约定；Qwen21 的 preset 无须使用它。`custom_sigmas` 仍按约定绕过 shift。

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
5. dummy 快照在 detach/broadcast 前过滤 `_` 字段，dummy forward 临时关闭缓存上下文，结束后删除临时字段。无需在每个 writer、RPC 或 rollout 输出安装过滤器。

缓存的作用域是一次采样轨迹中的条件分支与 variant。期间权重、条件、设备和 dtype 固定；不做内容 hash、LRU 或权重版本管理。
推理及 no-grad rollout 复用缓存；训练反传复用暂不设计。直接调用 adapter 的公开接口默认不保留缓存。

### microbatch

当前 Flux 的 `_txt_ids` / `_img_ids` 通过 `shared_cache_fields` 声明为布局相同即可共享的字段。通用 dense collator 从兼容样本取已有值，forward 后写回各逻辑样本。
Flux2 保存 `[1, N, 4]`，使用时展开到当前 B，避免交错采样导致 batch 大小变化后用错形状。

**K/V 依赖每个样本的条件，不能声明为共享字段。** 两个新 adapter 均按逻辑样本逐个 forward；`_kv_cache` 为空时执行 `extract`，有值时执行 `cached`。多个请求、CFG 和 variant 可以由 Executor 交错调度。
后续 dense 支持由 Qwen21 adapter 自己组装、拆分缓存，不在通用层预建递归 scatter 框架：

- 除已有 shape 检查外，要求 `image_pad_mask` 内容与完整 `img_shapes`（参考图 H/W、顺序及目标尺寸）一致，因为上游读取第一行布局。
- 全部未 prefill：一次 dense prefill，沿 B 拆分后保存到各 row。
- 全部已有缓存：临时沿 B 拼接每层 K/V，执行 dense decode。
- 混合阶段或布局不同：使用现有跨 rank 一致的逐样本回退。

batch=1 直接保存上游 cache。B>1 prefill 拆分时 clone，避免一条样本保留整个 batch 的存储；dense decode 的临时拼接会增加峰值显存，需要计入基准。

CPU 测试覆盖 Flux ID 的 B=2→1、CFG 隔离、variant 切回命中、完成/yield/异常/close 清理，以及 dummy 不传播运行时字段。新增小型真实 Qwen21/Flux2 Transformer 验证缓存开关的 float32 一致性、有/无参考图混排、CFG 分支隔离、反向传播，以及 Qwen21 meta 物化后的前向一致性。

上游明确说明 Qwen21 的缓存与完整前向在 BF16 下可能生成可见差异：两条注意力路径的归约顺序不同，逐层、逐步放大舍入差异。数值等价用 float32 检查，实际 BF16 样图同时保存开关两种结果，不承诺逐像素一致。

BF16、batch=1 下，按配置推算缓存成本为 `32层 × 2(K/V) × 4096 × 2字节`，即每个前缀 token 0.5 MiB。
一张经 resize 后为 1024² 的参考图产生 4096 个 latent token，其常驻 K/V 约 2 GiB，另加文本、激活和合批临时内存；这是张量容量估算，不是实测峰值。多参考图应同时测吞吐与显存。

## 训练与性能验证顺序

1. **依赖/组件**：更新后验证全仓库导入；对齐 encoder 特征与两类 mask、RGB/RGBA VAE encode/decode、latent pack/unpack、官方 sigma 网格。
2. **无缓存推理**：固定初始 latent、prompt、尺寸和 scheduler，逐步比较官方 pipeline 与 adapter 的 velocity/latent；覆盖 T2I、单参考图、多参考图、透明输出。仅相同 seed 不保证随机数布局一致。
3. **训练闭环**：验证 LoRA 梯度、一步 optimizer update、保存/重载与 FSDP。上游有 PEFT、gradient checkpointing 和 `_no_split_modules`，适配仓库机制有基础，但当前尚未做真模型训练验证，也未确认最优微调配方。
4. **缓存/批处理**：验证跨 timestep 等价、不同请求与 CFG/LoRA 分支隔离、异构布局回退、空 rank/dummy forward，以及 rollout/replay 的策略边界。
5. **性能**：基准先保留上游默认分段 SDPA；FlexAttention 作为独立优化。上游特别指出未编译的 flex 路径可能物化巨大的 attention 矩阵，不能只换 processor 就宣称更快。正式 GPU 验证应在 Slurm 作业中运行并监测利用率。

新 LoRA 可先覆盖 `attn.to_q/to_k/to_v/to_out.0`，再评估 `img_mlp.proj/out/gate_layer`；旧 Qwen 的 `add_*`、`txt_mlp` 和 `img_mod` 等目标不适用。
`QwenImage21Adapter._install_modules()` 在 CPU 上重建普通 complex64 RoPE 表，避免 meta 张量无法搬运或注册 buffer 后被 BF16 转换丢失相位；时间频率改为每次按配置重算 128 个 float32 常量，避免 `to_empty` 后非持久 buffer 未初始化。这不改变 checkpoint schema。
