## FLUX-like DiTs

### Preliminary

FLUX = InstructPix2Pix style post-training + Stable Diffusion 3.0 style architecture, with everything in self-attention context.

0. [2211.09800](https://arxiv.org/abs/2211.09800) InstructPix2Pix: Learning to Follow Image Editing Instructions
   - UC Berkeley · 2022-11 · [Code](https://github.com/timothybrooks/instruct-pix2pix) · 利用 GPT-3 与 SD 合成指令编辑数据，训练条件扩散模型实现自然语言驱动的图像编辑。
0. [2403.03206](https://arxiv.org/abs/2403.03206) [Stable Diffusion 3.0] Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
   - Stability AI · 2024-03 · [Code](https://github.com/Stability-AI/sd3-ref) · 提出 MMDiT 双流 Transformer 与改进的 Rectified Flow 采样，实现高分辨率文生图。

### FLUX and its Friends

Most implementations of FLUX-like DiTs can be found in the [`diffusers`](https://github.com/huggingface/diffusers) library.

It is said that Nano Banana belongs to this category.

0. [2505.22705](https://arxiv.org/abs/2505.22705) HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer
   - HiDream.ai · 2025-05 · [Code](https://github.com/HiDream-ai/HiDream-I1) · 17B 参数稀疏 MoE-DiT 结合流匹配，高效生成高质量图像的开源基础模型。
1. [2506.15742](https://arxiv.org/abs/2506.15742) FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
   - Black Forest Labs · 2025-06 · [Code](https://github.com/black-forest-labs/flux) · 基于流匹配的统一架构，通过序列拼接实现上下文图像生成与多轮一致性编辑。
2. [2508.02324](https://arxiv.org/pdf/2508.02324) Qwen-Image Technical Report
   - 阿里通义 · 2025-08 · [Code](https://github.com/QwenLM/Qwen-Image) · 20B 参数 MMDiT 基础模型，擅长复杂中英文文本渲染与高保真图像编辑。
3. [Release Blog](https://bfl.ai/blog/flux-2) [VAE Blog](https://bfl.ai/research/representation-comparison) FLUX.2: Frontier Visual Intelligence
   - Black Forest Labs · 2025-11 · [Code](https://github.com/black-forest-labs/flux2) · 32B 参数旗舰模型，结合 Mistral-3 VLM 与 Rectified Flow Transformer 实现统一生成与编辑。
4. [2511.22982](https://arxiv.org/abs/2511.22982) Ovis-Image Technical Report
   - 阿里 AIDC-AI · 2025-11 · [Code](https://github.com/AIDC-AI/Ovis-Image) · 7B 文生图模型，专攻高质量文字渲染，单卡可部署。
5. [2511.22699](https://arxiv.org/abs/2511.22699) Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer
   - 阿里通义 · 2025-11 · [Code](https://github.com/Tongyi-MAI/Z-Image) · 6B 单流 DiT 高效图像生成基础模型，支持中英双语。
6. [2512.07584](https://arxiv.org/abs/2512.07584) LongCat-Image Technical Report
   - 美团 LongCat · 2025-12 · [Code](https://github.com/meituan-longcat/LongCat-Image) · 6B 中英双语开源图像生成与编辑基础模型，强汉字渲染。
7. [2601.02358](https://arxiv.org/pdf/2601.02358) VINO: A Unified Visual Generator with Interleaved OmniModal Context
   - 快手 Kling · 2026-01 · [Code](https://github.com/SOTAMak1r/VINO-code) · 基于交错全模态上下文的统一图像/视频生成与编辑模型。
8. [2602.13344](https://arxiv.org/abs/2602.13344) FireRed-Image-Edit-1.0 Technical Report
   - 小红书 · 2026-02 · [Code](https://github.com/FireRedTeam/FireRed-Image-Edit) · 指令式图像编辑扩散 Transformer，开源 SOTA。
9. [Tech Report](https://joyai-image.s3.cn-north-1.jdcloud-oss.com/JoyAI-Image.pdf) JoyAI-Image: Awakening Spatial Intelligence in Unified Multimodal Understanding and Generation
   - 京东云 · N/A · [Code](https://github.com/jd-opensource/JoyAI-Image) · 8B MLLM + 16B MMDiT 统一多模态模型，主打空间智能。
10. [2603.28713](https://arxiv.org/abs/2603.28713) DreamLite: A Lightweight On-Device Unified Model for Image Generation and Editing [^1]
    - 字节跳动智能创作 · 2026-03 · [Code](https://github.com/ByteVisionLab/DreamLite) · 端侧轻量统一扩散模型，支持文生图与图像编辑。
11. [Tech Repport](https://github.com/xgen-universe/Capybara/blob/main/assets/docs/tech_report.pdf) CAPYBARA: A Unified Visual Creation Model [^2]
    - HKUST 等 · 2026-02 · [Code](https://github.com/xgen-universe/Capybara) · 统一视觉创作模型，覆盖图像/视频生成与指令编辑。

[^1]: This is not necessarily a DiT since it uses U-Net architecture. But it still benifits from the most of FLUX recipes.
[^2]: This one also unifies video generation and editing.

## Unified Diffusion Models

These models also utilize the diffusion transformer architecture like FLUX. They are categorized as unified diffusion models because they jointly train both the image generation and understanding tasks.

1. [2505.14683](https://arxiv.org/abs/2505.14683) [BAGEL]: Emerging Properties in Unified Multimodal Pretraining
   - 字节跳动 Seed · 2025-05 · [Code](https://github.com/ByteDance-Seed/Bagel) · 7B 激活参数 MoT 开源统一多模态理解与生成模型。
2. [2506.18871](https://arxiv.org/pdf/2506.18871) OmniGen2: Towards Instruction-Aligned Multimodal Generation
   - BAAI 等 · 2025-06 · [Code](https://github.com/VectorSpaceLab/OmniGen2) · 双解码路径统一多模态生成模型，支持指令对齐编辑。
3. [2509.23951](https://arxiv.org/abs/2509.23951) HunyuanImage 3.0 Technical Report
   - 腾讯混元 · 2025-09 · [Code](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) · 80B MoE 原生多模态自回归图像生成模型。
4. [Release Blog](https://z.ai/blog/glm-image) GLM-Image: Auto-regressive for Dense-knowledge and High-fidelity Image Generation
   - 智谱 Z.ai · 2025-11 · [Code](https://github.com/zai-org/GLM-Image) · 16B 自回归图像生成模型，强于密集知识与文本渲染。
5. [2603.09877](https://arxiv.org/abs/2603.09877) InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing
   - 上海 AI Lab OpenGVLab · 2026-03 · [Code](https://github.com/OpenGVLab/InternVL-U) · 4B 参数统一多模态模型，融合理解、推理、生成与编辑能力。

## AR or Other Architectures

It is said that GPT-Image-2 belongs to this category, which is an autoregressive VQ-VAE model with a lightweight diffusion head.

1. [2404.02905](https://arxiv.org/abs/2404.02905) [VAR]: Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
   - 字节跳动 & 北大 · 2024-04 · [Code](https://github.com/FoundationVision/VAR) · 用"下一尺度预测"的视觉自回归建模，让 AR 首次超越扩散模型。
2. [2504.17761](https://arxiv.org/abs/2504.17761) Step1X-Edit: A Practical Framework for General Image Editing
   - 阶跃星辰 · 2025-04 · [Code](https://github.com/stepfun-ai/Step1X-Edit) · 结合 MLLM 与 DiT 的开源通用图像编辑框架，对标 GPT-4o。
3. [2602.04770](https://arxiv.org/abs/2602.04770) Generative Modeling via Drifting
   - MIT（何恺明组）· 2026-02 · [Code](https://github.com/lambertae/drifting) · 提出漂移模型范式，训练时演化推送分布以实现一步生成 SOTA。
4. [2603.27538](https://arxiv.org/pdf/2603.27538) LongCat-Next: Lexicalizing Modalities as Discrete Tokens
   - 美团 LongCat · 2026-03 · [Code](https://github.com/meituan-longcat/LongCat-Next) · 原生离散多模态模型，将图音文统一为 NTP 离散 token。

## Agent / Pipelines / Applications

1. [2509.04338](https://arxiv.org/abs/2509.04338) From Editor to Dense Geometry Estimator
   - 北交大 / 阿里 AMAP / 重邮 / NTU · 2025-09 · [Code](https://github.com/AMAP-ML/FE2E) · 将图像编辑器微调为零样本稠密深度与法线估计器。
2. [2601.18428](https://arxiv.org/abs/2601.18428) Collaposer: Transforming Photo Collections into Visual Assets for Storytelling with Collages
   - HKUST · 2026-01 · N/A · 自动从照片集中分割提取素材以辅助拼贴叙事的工具。
3. [2601.23265](https://arxiv.org/abs/2601.23265) PaperBanana: Automating Academic Illustration for AI Scientists
   - 北大 & Google Research · 2026-01 · [Code](https://github.com/dwzhu-pku/PaperBanana) · 多智能体框架自动生成学术论文配图与示意图。
4. [2602.09084](https://arxiv.org/abs/2602.09084) Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling
   - 德州农工大学 TACO · 2026-02 · [Code](https://github.com/taco-group/agent-banana) · 规划-执行智能体实现 4K 多轮高保真图像编辑。
5. [2603.28767](https://arxiv.org/abs/2603.28767) Gen-Searcher: Reinforcing Agentic Search for Image Generation
   - 港中文 MMLab / UCLA / UCB · 2026-03 · [Code](https://github.com/tulerfeng/Gen-Searcher) · 强化学习训练图像生成智能体先搜索后作画。
6. [2604.19587](https://arxiv.org/abs/2604.19587) SmartPhotoCrafter: Unified Reasoning, Generation and Optimization for Automatic Photographic Image Editing
   - vivo 蓝厂影像 · 2026-04 · N/A · 统一推理、生成与优化的全自动摄影后期修图模型。
7. [2604.20329](https://arxiv.org/abs/2604.20329) [Vision Banana]: Image Generators are Generalist Vision Learners
   - Google DeepMind · 2026-04 · N/A · 指令微调 Nano Banana Pro 使图像生成器成为通用视觉大模型。
