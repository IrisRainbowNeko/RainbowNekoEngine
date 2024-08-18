# RainbowNeko Engine


## Introduction
RainbowNeko Engine is a toolbox for pytorch based neural network training and inference. Multiple tasks and training strategies are built-in and highly expandable.


## Install

1. Install [pytorch](https://pytorch.org/)

2. Install from source:
```bash
git clone https://github.com/IrisRainbowNeko/RainbowNekoEngine.git
cd RainbowNekoEngine
pip install -e .
# Modified based on this project or start a new project and make initialization
nekoinit
```

3. To use xFormers to reduce VRAM usage and accelerate training:
```bash
# use conda
conda install xformers -c xformers

# use pip
pip install xformers>=0.0.17
```

## User guidance

### Training

Training scripts based on 🤗 Accelerate or Colossal-AI are provided.
+ For 🤗 Accelerate, you may need to [configure the environment](https://github.com/huggingface/accelerate/tree/main#launching-script) before launching the scripts.
+ For Colossal-AI, you can use [torchrun](https://pytorch.org/docs/stable/elastic/run.html) to launch the scripts.

```yaml
# with Accelerate
neko_train --cfg cfgs/train/cfg_file.yaml
# with Accelerate and only one GPU
neko_train_1gpu --cfg cfgs/train/cfg_file.yaml
```

### Inference
TODO

### Tutorials
TODO

## Contributing

You are welcome to contribute more models and features to this toolbox!
