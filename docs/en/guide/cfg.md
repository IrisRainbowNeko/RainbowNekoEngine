# Configuration File Explanation

## Basic Format

```{tip}
The configuration files for the RainbowNeko Engine support two formats: **Python** and **YAML**. It is recommended to use the **Python** format due to its higher flexibility, simplicity, and better readability.
```

### Python Format

The Python format allows the use of full Python syntax in configuration files, enabling function and class calls within the configuration. For example:

```python
from functools import partial

from cfgs.py.train.classify import multi_class
from rainbowneko.parser import make_base
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data.handler import MixUPHandler, HandlerChain
from rainbowneko.train.loss import LossContainer, SoftCELoss

num_classes = 10
multi_class.num_classes = num_classes


def make_cfg():
    dict(
        _base_=make_base(multi_class) + [],

        train=dict(
            loss=LossContainer(loss=SoftCELoss()),
            metrics=None,
        ),

        data_train=dict(
            dataset1=BaseDataset(
                batch_handler=HandlerChain(handlers=dict(
                    mixup=MixUPHandler(num_classes=num_classes)
                ))
            )
        ),
    )
```

Configurations must be defined within a `make_cfg` function as a `dict`. Full Python syntax is supported in configurations, including function calls and operations.

````{note}
The configuration function is not executed directly. Instead, it is parsed by the parser using AST (Abstract Syntax Tree), converting all `call` operations into `dict` and `list`. After parsing, the framework instantiates components as needed.

For example:
```python
dict(
    layer=Linear(4, 4, bias=False)
)
```

During parsing, this will be automatically translated into:
```python
dict(
    layer=dict(_target_=Linear, _args_=[4, 4], bias=False)
)
```
````

#### Using `partial`

Some modules in the configuration require the use of `partial` because additional parameters need to be passed when using them. This can be achieved in two ways:

```python
optimizer=partial(torch.optim.AdamW, weight_decay=5e-4)
# Automatically converted by the parser
optimizer=torch.optim.AdamW(_partial_=True, weight_decay=5e-4)
```

#### Configuration Functions

### YAML Format

In YAML-format configuration files, when referencing a class or function, you must specify the full path. For example:

```yaml
_base_:
  - cfgs/yaml/train/classify/multi_class.yaml

num_classes: 10

train:
  loss:
    _target_: rainbowneko.train.loss.LossContainer
    loss:
      _target_: rainbowneko.train.loss.SoftCELoss
    metrics: null
    
data_train:
  dataset1:
    _target_: rainbowneko.train.data.BaseDataset
    batch_handler:
      _target_: rainbowneko.train.data.handler.HandlerChain
      handlers:
        mixup:
          _target_: rainbowneko.train.data.handler.MixUPHandler
          num_classes: ${num_classes} # Reference configuration value
```

## Inheritance

Configuration files can inherit from others. In Python configuration files, this is achieved by importing another configuration file and specifying it in `_base_`.

```python
from cfgs.py.train.classify import multi_class
from rainbowneko.parser import make_base

dict(
    _base_=make_base(multi_class) + [],
    ...
)
```

For example, inheriting the `multi_class` configuration file allows you to automatically retrieve its path through the `make_base` function.

```{note}
The `_base_=make_base(multi_class) + []` construct ensures that `make_base` executes directly. Operations like `+`, `-`, `*`, `/` around call nodes prevent them from being converted into `dict` or `list` by the parser.
```

Parameters defined in a child configuration file override those in the parent file. For nested configurations, only explicitly defined parameters are replaced; entire dictionaries or calls are not replaced unless specified.

For example, if the parent configuration defines `data_train` as follows:

```python
dict(
    dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
        source=dict(
            data_source1=IndexSource(
                data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True)
            ),
        ),
        handler=HandlerChain(handlers=dict(
            load=LoadImageHandler(),
            bucket=FixedBucket.handler,
            image=ImageHandler(transform=T.Compose([
                    T.RandomCrop(size=32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]),
            )
        )),
        bucket=FixedBucket(target_size=32),
    )
)
```

To modify only the dataset path in a child configuration file:

```python
dict(
    dataset1=partial(BaseDataset,
        source=dict(
            data_source1=IndexSource(
                data=torchvision.datasets.cifar.CIFAR10(root='data path')
            ),
        ),
    )
)
```

This modifies only the `root` parameter for CIFAR10 without affecting other parameters like `handler` and `bucket`.

````{tip}
Since the parser converts calls into dictionaries during inheritance, parameters can also be modified as follows:

```python
dict(
    dataset1=dict(
        source=dict(
            data_source1=dict(
                data=dict(root='data path')
            ),
        ),
    )
)
```
Here, `IndexSource()` is equivalent to `dict(_target_=IndexSource)`.
````

### Full Replacement

To completely replace a node from a parent configuration file without retaining any part of it:

```python
dataset1=partial(BaseDataset,
    _replace_=True,
    ...
)
```

### Deletion

To delete a node from a parent configuration file:

```python
dict(
    dataset1='---', # Deletes the dataset1 node
    dataset_new=...
)
```

## Referencing Other Configurations

You can reference configurations from other nodes within a single node:

```python
train=dict(
    train_epochs=100,
)
epochs='${train.train_epochs}' # References train_epochs from train node.
```

Relative paths can also be used for references:

```python
model=dict(
    wrapper=DistillationWrapper(_partial_=True, _replace_=True,
        model_teacher=load_resnet(torchvision.models.resnet18()),
        model_student='${.model_teacher}', # References model_teacher at the same level.
        ...
    )
),
```