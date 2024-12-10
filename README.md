# RainbowNeko Engine

[![PyPI](https://img.shields.io/pypi/v/rainbowneko)](https://pypi.org/project/rainbowneko/)
[![GitHub stars](https://img.shields.io/github/stars/IrisRainbowNeko/RainbowNekoEngine)](https://github.com/IrisRainbowNeko/RainbowNekoEngine/stargazers)
[![GitHub license](https://img.shields.io/github/license/IrisRainbowNeko/RainbowNekoEngine)](https://github.com/IrisRainbowNeko/RainbowNekoEngine/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/IrisRainbowNeko/RainbowNekoEngine/branch/main/graph/badge.svg)](https://codecov.io/gh/IrisRainbowNeko/RainbowNekoEngine)
[![open issues](https://isitmaintained.com/badge/open/IrisRainbowNeko/RainbowNekoEngine.svg)](https://github.com/IrisRainbowNeko/RainbowNekoEngine/issues)


[📘English document](https://rainbownekoengine.readthedocs.io/en/latest/)
[📘中文文档](https://rainbownekoengine.readthedocs.io/zh-cn/latest/)

## Introduction
RainbowNeko Engine is a toolbox for pytorch based neural network training and inference. Multiple tasks and training strategies are built-in and highly expandable.


## Install

1. Install [pytorch](https://pytorch.org/)

2. Install with pip:
```bash
pip install rainbowneko
```

or
Install from source:
```bash
git clone https://github.com/IrisRainbowNeko/RainbowNekoEngine.git
cd RainbowNekoEngine
pip install -e .
```

3. To use xFormers to reduce VRAM usage and accelerate training:
```bash
# use conda
conda install xformers -c xformers

# use pip
pip install xformers>=0.0.17
```

## User guidance

### Start a new project
```bash
mkdir my_project
cd my_project
# Modified based on this project or start a new project and make initialization
nekoinit
```

### Training

Training scripts based on 🤗 Accelerate or Colossal-AI are provided.
+ For 🤗 Accelerate, you may need to [configure the environment](https://github.com/huggingface/accelerate/tree/main#launching-script) before launching the scripts.
+ For Colossal-AI, you can use [torchrun](https://pytorch.org/docs/stable/elastic/run.html) to launch the scripts.

```yaml
# with Accelerate
neko_train --cfg cfgs/train/cfg_file.py
# with Accelerate and only one GPU
neko_train_1gpu --cfg cfgs/train/cfg_file.py
```

### Inference
RainbowNeko Engine inference with workflow configuration file.

```yaml
neko_run --cfg cfgs/infer/cfg_file.py
```

### Tutorials

[📘English document](https://rainbownekoengine.readthedocs.io/en/latest/)
[📘中文文档](https://rainbownekoengine.readthedocs.io/zh-cn/latest/)

## Contributing

You are welcome to contribute more models and features to this toolbox!
