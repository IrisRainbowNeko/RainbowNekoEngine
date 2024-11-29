# Introduction

RainbowNeko Engine is a general-purpose deep learning framework based on PyTorch. It is designed to provide a simple and user-friendly deep learning toolkit, enabling users to quickly build, train, and deploy deep learning models.

RainbowNeko Engine uses `python` and `yaml` configuration files for experiment management. All parameters required during the training or inference phases can be configured through these files. Users can define datasets, models, optimizers, loss functions, and other components using straightforward configuration files. Training, evaluation, and inference are all performed based on these configurations. This approach is convenient, highly readable, and facilitates reproducibility.

```{note}
For detailed explanations of the configuration files, refer to [Configuration File Details](./cfg.md).
```

## Features

* Layered training configuration
* Built-in LoRA module
* Model plugin system
* Aspect Ratio Bucket (ARB) with automatic clustering
* Integration with [🤗 Accelerate](https://github.com/huggingface/accelerate)
* Training acceleration with DeepSpeed
* Modular dataset architecture

### Python-Style Configuration Files

RainbowNeko Engine supports configuration files written in a Python-like syntax. This allows users to call functions and classes directly within the configuration file, with function parameters inheritable from parent configuration files. The framework automatically handles the formatting of these configuration files.

For example, consider the following configuration file:
```python
dict(
    layer=Linear(in_features=4, out_features=4)
)
```
During parsing, this will be automatically translated into:
```python
dict(
    layer=dict(_target_=Linear, in_features=4, out_features=4)
)
```
After parsing, the framework will instantiate the components accordingly. This means users can write configuration files using familiar Python syntax.

### Data Flow Control

RainbowNeko Engine allows users to configure data flow directly within the configuration file—from data loading to loss computation and evaluation metrics calculation—providing high flexibility. Data transmission paths between modules can be fully controlled using only the configuration file.