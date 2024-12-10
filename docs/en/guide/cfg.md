# Configuration File Explanation

## Basic Format

```{tip}
The configuration files for the RainbowNeko Engine support both **Python** and **YAML** formats. It is recommended to use the **Python** format due to its higher flexibility, simplicity, ease of use, and better readability.
```

### Python Format

Configuration files in Python format support full Python syntax, allowing for function and class calls within the configuration. For example:

```python
from functools import partial

from cfgs.py.train.classify import multi_class
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data.handler import MixUPHandler, HandlerChain
from rainbowneko.train.loss import LossContainer, SoftCELoss

num_classes = 10
multi_class.num_classes = num_classes


def make_cfg():
    return dict(
        _base_=[multi_class],

        train=dict(
            loss=LossContainer(loss=SoftCELoss()),
            metrics=None,
        ),

        data_train=dict(
            dataset1=BaseDataset(
                batch_handler=HandlerChain(
                    mixup=MixUPHandler(num_classes=num_classes)
                )
            )
        ),
    )
```

The configuration should be defined within a `make_cfg` function that returns a `dict`. Full Python syntax is supported in the configuration, including function calls and operations.

````{note}
The configuration function is not executed directly. Instead, it is parsed by an interpreter using AST (Abstract Syntax Tree), which converts all `call` operations into `dict` and `list`. After parsing, the framework instantiates them where necessary.

For example:
```python
dict(
    layer=Linear(4, 4, bias=False)
)
```
During parsing, it will be automatically translated into:
```python
dict(
    layer=dict(_target_=Linear, _args_=[4, 4], bias=False)
)
```
````

```{note}
Operations such as `+-*/` on both sides of a `call` node will not be converted into `dict` or `list` by the parser; they will be executed directly.
```

#### Using `partial`

Some modules in the configuration may require additional parameters during use. These can be defined using `partial`, which can be implemented in two ways:

```python
optimizer = partial(torch.optim.AdamW, weight_decay=5e-4)
# Automatically converted by the parser
optimizer = torch.optim.AdamW(_partial_=True, weight_decay=5e-4)
```

#### Configuration Function

### YAML Format

In YAML format configuration files, when referencing a class or function, you must provide its full path. For example:

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
      mixup:
        _target_: rainbowneko.train.data.handler.MixUPHandler
        num_classes: ${num_classes} # Reference to another configuration parameter
```

## Inheritance

Configuration files can inherit from others. For example, in Python configuration files, you can inherit another file’s settings by importing it and specifying it in `_base_`:

```python
from cfgs.py.train.classify import multi_class

dict(
    _base_=[multi_class],
    ...
)
```

Here, inheriting the `multi_class` configuration file automatically includes its content.

Parameters defined in the current configuration override those from the parent file. For nested configurations, only inner parameters are replaced; the entire `dict` or call is not replaced.

For instance, if the parent file’s `data_train` has this structure:

```python
dict(
    dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
        source=dict(
            data_source1=IndexSource(
                data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True)
            ),
        ),
        handler=HandlerChain(
            load=LoadImageHandler(),
            bucket=FixedBucket.handler,
            image=ImageHandler(transform=T.Compose([
                    T.RandomCrop(size=32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]),
            )
        ),
        bucket=FixedBucket(target_size=32),
    )
)
```

You can modify just the dataset path in a child file like this:

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

This only modifies the `root` parameter of `CIFAR10`, leaving other parameters unchanged. The `handler` and `bucket` parameters within `dataset1` remain unaltered.

````{tip}
Since the parser converts calls into dictionaries during inheritance, you can modify parameters like this:

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

Here, calling `IndexSource()` is equivalent to writing `dict(_target_=IndexSource)`.
````

### Complete Replacement

To completely replace a parent file’s node without retaining any part of it:

```python
dataset1 = partial(BaseDataset,
    _replace_=True,
    ...
)
```

### Deletion

To delete a node from a parent file:

```python
dict(
    dataset1='---', # Deletes the dataset1 node
    dataset_new=...
)
```

## Referencing Other Configurations

A node can reference another node’s parameter. For example:

```python
train=dict(
    train_epochs=100,
)
epochs='${train.train_epochs}' # Reference to train's train_epochs parameter
```

You can also use relative paths for references:

```python
model=dict(
    wrapper=DistillationWrapper(_partial_=True, _replace_=True,
        model_teacher=load_resnet(torchvision.models.resnet18()),
        model_student='${.model_teacher}', # Reference to sibling node model_teacher
        ...
    )
),
```