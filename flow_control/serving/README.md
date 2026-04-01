# flow_control.serving

flow_control.serving 模块通过 Gradio 提供了一个简单的 Web 界面，方便快速测试和展示 DiT 模型训练的效果。

flow_control.serving 模块实现：

1. 允许用户在 Gradio 界面上上传输入需要的文本和图像等数据，并可以尽可能多地修改 Pydantic 配置中影响模型生成效果的参数。
2. 灵活的检查点加载功能。既可以直接从 Hugging Face Hub 加载预训练模型，也可以从本地加载 DCP Seed 检查点。还能在此基础上加载训练保存的检查点。
3. 动态检查点加载功能。对于每条请求，动态加载需要的检查点和辅助模型权重，跳过已加载的模型，提升响应速度。
4. 支持简单 Offloading：目前主要面向在多卡机器上运行，支持将 Processor 和 DiT 模型加载到不同的 GPU 上，对于一张卡放不下的模型，可以用 Accelerate 的 device map 机制做 Model Parallel。对于单卡机器，支持在 DiT 运算时，将 Processor 模型 Offload 到 CPU 上，节省 GPU 显存。

不在计划中：

1. 大批量，高效率的推理接口。Gradio 只用负责一次处理一个请求。请考虑使用离线批量推理脚本。
2. 更复杂的 CPU Offloading 机制，如对 DiT 权重的动态 Offloading 和分层 Offloading。