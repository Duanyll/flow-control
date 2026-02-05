# 简要说明

总的来说，利用 diffusers 的 model 层，但不用 pipeline 层，自己写训练循环和采样逻辑

- adapters: 训练时需要每一步做的事情，与模型架构有关的逻辑
- datasets: 如何从硬盘上读写数据
- processors: 预处理数据，不需要训练中每一步都做的事情，可以离线做完存盘
- samplers: Diffusion / Flow 采样算法。实际上是 diffusers 里的 sampler + scheduler 的结合体
- scripts: 命令行入口点
- training: 训练循环（实际上也包含了 HsdpInference 相关的逻辑）
- utils: 各种工具函数

总体原则：老套的 OOP（大量利用多态，DRY 原则），反对 Huggingface 的 single file policy 思想。

避免使用函数参数（尤其是意义不明的 kwargs）传递整个程序生命周期中都不变的配置，只传递数据，将配置放在 Pydantic 类的属性里。多数类都不持有非 Pydantic 配置的状态，多数算法都实现在类的成员方法中，可以方便地被继承和重载。

尽量详细地写类型注解，或者让设计模式对类型检查友好，确保通过 pyright 的类型检查。目前几个不得不 Any 或者 `# type: ignore` 的地方：

- 来自 diffusers 和 transformers 的类型。这两个库的类型注解难以被正确推导
- adapters 和 processors 内部对 batch 分别用 TypedDict 标注了类型，但在训练循环和采样循环中传递时，使用了 dict 或 Any，因为 Python 对泛型和继承的支持有限，很难正确的把 dataset， adapter，processor，training loop 之间的类型传递都标注出来
- 多态子类的类型几乎都不协变，这是刻意的设计（原因其实也在于 Python 泛型支持的局限性）。需要移除一些类型检查规则（`reportIncompatibleMethodOverride`）来允许子类的方法接受比父类更具体的参数