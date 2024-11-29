# 训练你的第一个模型

RainbowNeko Engine使用配置文件来配置训练中用到的各种参数和模块。在这个例子中，我们将使用一个多类别分类任务的配置文件来用`CIFAR-10`训练一个简单的`resnet`模型。

RainbowNeko Engine支持`python`和`yaml`两种格式的配置文件，这里我们使用`python`格式的配置文件作为例子。`python`格式的配置文件完全由python
语法编写，可以更加灵活地定义模型和数据，也有更高的可读性，使用起来更加清晰和方便。

## 初始化工程
```{attention}
RainbowNeko Engine应该被当做一个库来使用，而不是直接在它的文件夹中进行训练或推理。
```

新建一个文件夹:
```bash
mkdir my_first_project
cd my_first_project
```

初始化工程，会在当前文件夹自动创建必要的配置文件
```bash
nekoinit
```

## 开始训练

::::{tab-set}
:::{tab-item} 单GPU训练
运行下面的命令，指定配置文件，便可以开始训练模型啦。所有数据和模型的定义都在配置文件中。
```bash
# Train with Accelerate and only one GPU
neko_train_1gpu --cfg cfgs/py/train/classify/multi_class.py
```
:::

:::{tab-item} 多GPU训练
多GPU训练需要在`cfgs/launcher/multi.yaml`中指定训练用到的GPU id和GPU数量，然后运行:
```bash
# Train with Accelerate and multiple GPUs
neko_train --cfg cfgs/py/train/classify/multi_class.py
```

````{tip}
你还可以复制一份`cfgs/launcher/multi.yaml`配置文件，修改其中的参数，之后通过`--launch_cfg`参数指定新的配置文件。
>   ```bash
>   neko_train --launch_cfg cfgs/launcher/multi_2.yaml --cfg cfgs/py/train/classify/multi_class.py
>   ```
````
:::
::::

## 调整训练参数
这一节会介绍如何在配置文件中调整训练相关的参数。

```{note}
配置文件中，配置定义在`make_cfg`函数中，由`dict`、`list`和各种类与函数的调用构成。
```

### 调整学习率
在配置文件中`model_part`定义了模型各层的训练参数，调整`lr`可以改变学习率:
```python
model_part=CfgWDModelParser([
    dict(
        lr=2e-4, # 设置学习率为2e-4
        layers=[''],  # 训练所有层
    )
]),
```

```{tip}
通过`model_part`可以单独设置每一层的学习率等参数，详细说明见 [分层训练](../train_adv/layer_train.md)
```

### 调整其他参数
在`train`这一项中，可以定义一些训练常用的参数:
```python
train=dict(
    train_steps=1000, # 训练总步数
    train_epochs=100, # 训练总epochs，这个不为None则会忽略train_steps
    workers=2, # 读取数据的进程数
    max_grad_norm=None, # 梯度裁剪
    save_step=2000, # 保存模型步数间隔
    gradient_accumulation_steps=1, # 梯度累积
)
```

## 修改模型
在配置文件中，`model`部分定义了训练使用的模型。
```python
def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model=dict(
    name='cifar-resnet18', # 模型的名字，保存时使用
    wrapper=partial(SingleWrapper, model=load_resnet()) # 模型结构，这里通过load_resnet()定义
),
```

```{note}
RainbowNeko Engine中模型都需要用Wrapper包装起来，对于只有一个模型，且只有一条数据流的任务，可以使用`SingleWrapper`。
```

## 调整batch size

`batch size`定义在数据配置部分`data_train`中，这里将数据配置放入一个单独的配置函数`cfg_data()`中。修改`cfg_data()`中的`batch_size`便可以调整。

```{tip}
由`@neko_cfg`装饰器装饰的函数，都会变成配置函数。在配置文件解析时，会直接将函数的配置内容放在调用它的位置。
```

## 更改数据路径
在数据配置中`dataset`的`source`定义了数据的来源，修改`data_source1`中CIFAR10的`root`可以修改数据路径。

```{note}
更多关于数据集的配置，可以查看[数据集配置](./data.md)
```