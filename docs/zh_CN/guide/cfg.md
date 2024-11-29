# 配置文件详解

## 基本格式

```{tip}
RainbowNeko Engine的配置文件有**python**和**yaml**两种格式。推荐使用**python**格式，灵活度更高，简单易用，可读性也更强。
```

### python格式

python格式的配置文件支持完整的python语法，可以在配置中进行函数和类的调用。比如下面的例子:
```python
from functools import partial

from cfgs.py.train.classify import multi_class
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data.handler import MixUPHandler, HandlerChain
from rainbowneko.train.loss import LossContainer, SoftCELoss

num_classes = 10
multi_class.num_classes = num_classes


def make_cfg():
    dict(
        _base_=[multi_class],

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

配置需要定义在`make_cfg`函数中，定义一个`dict`。在配置中支持完整的python语法，包括函数调用，运算等。

````{note}
配置函数不会直接执行，会由解析器使用ast(抽象句法树)进行解析，将所有的`call(调用)`操作都转换成`dict`和`list`。在解析之后，会由框架在需要的地方进行实例化。
比如:
```python
dict(
    layer=Linear(4, 4, bias=False)
)
```
在解析阶段，会自动翻译成:
```python
dict(
    layer=dict(_target_=Linear, _args_=[4,4], bias=False)
)
```

````

```{note}
`+-*/`等运算操作左右的`call`节点，不会被解析器转换成`dict`和`list`，会直接执行。
```


#### partial使用

在配置中，有一些模块需要使用`partial`定义，因为在使用该模块时需要传入额外的参数。通过以下两种方式都可以实现:
```python
optimizer=partial(torch.optim.AdamW, weight_decay=5e-4)
# 由解析器自动转换
optimizer=torch.optim.AdamW(_partial_=True, weight_decay=5e-4)
```

#### 配置函数


### yaml格式
在yaml格式的配置文件中，当需要引用某个类或函数时，需要写完整的路径。比如:
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
          num_classes: ${num_classes} # 引用配置
```

## 继承
配置文件可以继承，比如在python配置文件中，通过`import`导入配置文件，在`_base_`中配置，就可以继承另一个配置文件的配置。
```python
from cfgs.py.train.classify import multi_class

dict(
    _base_=[multi_class],
    ...
)
```
比如这里继承`multi_class`这个配置文件，可以自动获取`multi_class`的内容并继承。

在配置中定义的参数，会覆盖父配置文件的内容。对于嵌套的配置，只会替换内层定义的参数，不会把`dict`或者调用整个替换。

比如父配置文件的`data_train`是下面的配置:
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
            bucket=FixedBucket.handler, # bucket 会自带一些处理模块
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
如果要修改数据集路径，则可以在子配置文件中这样定义:
```python
dict(
    dataset1=partial(BaseDataset
        source=dict(
            data_source1=IndexSource(
                data=torchvision.datasets.cifar.CIFAR10(root='data path')
            ),
        ),
    )
)
```
这样只会修改`CIFAR10`的`root`参数，其他参数不会被修改。`dataset1`中的`handler`和`bucket`参数不会被修改。

````{tip}
由于解析器会把调用转换成dict，所以在继承时，可以这样修改参数:

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
    
`IndexSource()`等价于`dict(_target_=IndexSource)`
````

### 全部替换
如果要全部替换父配置文件某个节点的内容，不保留父配置文件的部分，可以添加`_replace_`:
```python
dataset1=partial(BaseDataset
    _replace_=True,
    ...
)
```

### 删除
如果要删除父配置文件的一个节点，可以使用`---`: 
```python
dict(
    dataset1='---', # 删除dataset1节点
    dataset_new=...
)
```

## 引用其他配置
在一个节点中可以引用其他节点的配置，比如:
```python
train=dict(
    train_epochs=100,
)
epochs='${train.train_epochs}' # 引用train节点的train_epochs
```

也可以使用相对路径进行引用:
```python
model=dict(
    wrapper=DistillationWrapper(_partial_=True, _replace_=True,
        model_teacher=load_resnet(torchvision.models.resnet18()),
        model_student='${.model_teacher}', # 引用同级节点的model_teacher
        ...
    )
),
```