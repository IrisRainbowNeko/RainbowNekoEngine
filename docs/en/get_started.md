# Getting Started

## Python Installation

### Download from the Official Website
1. Download [Python](https://www.python.org/downloads/) and install it.

### Install via Conda (Recommended)
1. Download [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
2. Open Miniconda and create a Python environment:
   ```bash
   conda create --name neko python=3.11 -y
   conda activate neko
   ```

## PyTorch Installation
Install PyTorch:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## RainbowNeko Engine Installation

::::{tab-set}
:::{tab-item} Install from Python Package
```bash
pip install rainbowneko
```
:::

:::{tab-item} Install from Source Code
Download and install:
```bash
git clone https://github.com/IrisRainbowNeko/RainbowNekoEngine.git
cd RainbowNekoEngine
pip install -e .
```
:::
::::

### Optional Installations

````{note}
Use xformers to reduce memory usage and accelerate training. Find the version corresponding to your PyTorch version on the [xformers releases](https://github.com/facebookresearch/xformers/releases) page, then install it:
```bash
pip3 install -U xformers==<version> --index-url https://download.pytorch.org/whl/<cuda_version> 
# Replace <cuda_version> with your CUDA version (e.g., cuda12.4 corresponds to cu124).
```

To monitor the training process with TensorBoard:
>   ```bash
>   pip install tensorboard
>   ```

To monitor the training process with WandB:
>   ```bash
>   pip install wandb
>   ```
````