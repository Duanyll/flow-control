# 简要说明

总的来说，利用 diffusers 的 model 层，但不用 pipeline 层，自己写训练循环和采样逻辑

- adapters: 训练时需要每一步做的事情，与模型架构有关的逻辑
- datasets: 如何从硬盘上读写数据
- processors: 预处理数据，不需要训练中每一步都做的事情，可以离线做完存盘
- samplers: Diffusion / Flow 采样方法。实际上是 diffusers 里的 sampler + scheduler，但我不喜欢 scheduler 这一层抽象
- scripts: 入口点脚本
- training: 训练循环
- utils: 各种工具函数