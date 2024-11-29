# 开始使用

## python安装

### 从官网下载
1. 下载[python](https://www.python.org/downloads/)并安装

### 通过conda安装 (推荐)
1. 下载[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
2. 打开Miniconda并创建python环境
```bash
conda create --name neko python=3.11 -y
conda activate neko
```

## pytorch安装
安装pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## RainbowNeko Engine安装

::::{tab-set}
:::{tab-item} 从python包安装
```bash
pip install rainbowneko
```
:::

:::{tab-item} 从源码安装
下载并安装
```bash
git clone https://github.com/IrisRainbowNeko/RainbowNekoEngine.git
cd RainbowNekoEngine
pip install -e .
```
:::
::::

### 可选安装
````{note}
使用xformers减少显存使用并加速训练。在 [xformers releases](https://github.com/facebookresearch/xformers/releases) 页面找到和你pytorch版本对应的版本号，然后安装:
```bash
pip3 install -U xformers==版本号 --index-url https://download.pytorch.org/whl/"cuda版本(比如cuda12.4对应cu124)"
```

使用tensorboard监控训练过程:
>   ```bash
>   pip install tensorboard
>   ```

使用wandb监控训练过程:
>   ```bash
>   pip install wandb
>   ```
````