# 框架介绍

RainbowNeko Engine是一个基于PyTorch的通用深度学习框架，旨在提供一个简单易用的深度学习工具包，帮助用户快速搭建、训练和部署深度学习模型。

RainbowNeko Engine基于`python`和`yaml
`格式的配置文件进行实验管理，训练或推理阶段用到的所有参数都可以通过配置文件进行配置。用户可以通过简单的配置文件定义数据集、模型、优化器、损失函数等各种内容，基于配置文件进行训练、评估和推理。使用方便，可读性高，便于复现。

```{note}
配置文件的详细说明见 [配置文件详解](./cfg.md)。
```

## 特性

* 分层训练配置
* 内置lora模块
* 模型插件系统
* 带自动聚类的Aspect Ratio Bucket (ARB)
* [🤗 Accelerate](https://github.com/huggingface/accelerate)
* Deepspeed训练加速
* 模块化数据集架构

### python格式的配置文件
RainbowNeko Engine支持由python语法习惯编写的配置文件，可以在配置中进行函数和类的调用，并且函数参数可以从父配置文件继承。框架会自动处理配置文件的格式。

比如下面的配置文件:
```python
dict(
    layer=Linear(in_features=4, out_features=4)
)
```
在解析阶段，会自动翻译成:
```python
dict(
    layer=dict(_target_=Linear, in_features=4, out_features=4)
)
```
在解析之后，会由框架进行实例化。所以用户可以直接用python语法习惯编写配置文件。

### 数据流控制
RainbowNeko Engine可以直接在配置文件中配置数据流，从数据读取到loss和评估指标的计算，灵活度较高。只通过配置文件就能控制数据在每个模块之间的传递路径。

