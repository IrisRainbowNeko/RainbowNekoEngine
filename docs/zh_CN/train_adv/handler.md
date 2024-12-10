# 数据处理高级配置

## 处理链

通过`HandlerChain`可以串行的组合多个数据处理模块，每个模块的输出覆盖输入的内容后，会作为下一个模块的输入。

```{image} ../../imgs/handler_chain.svg
:alt: Select Parameters
:width: 500px
:align: center
```

比如常见的图像处理:
```python
from rainbowneko.train.data.handler import HandlerChain, ImageHandler, LoadImageHandler

handler=HandlerChain(
    load=LoadImageHandler(),
    bucket=FixedBucket.handler, # bucket 会自带一些处理模块
    image=ImageHandler(transform=T.Compose([
            T.RandomCrop(size=32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]),
    )
),
```

## 处理组

通过`HandlerChain`可以并行的组合多个数据处理模块，数据输入到每个模块，并将它们的输出加起来作为输出。

```{image} ../../imgs/handler_group.svg
:alt: Select Parameters
:width: 400px
:align: center
```

比如读取图像进行不同的处理，放入不同的变量中:
```python
from rainbowneko.train.data.handler import HandlerGroup, HandlerChain, ImageHandler, LoadImageHandler

handler=HandlerChain(
    load=LoadImageHandler(),
    image=HandlerGroup(
        weak=ImageHandler(..., key_map_out=('image -> image_weak',)),
        strong=ImageHandler(..., key_map_out=('image -> image_strong',)),
    )
),
```

## 随机种子同步处理
在某些场景中，会需要多个处理模块使用相同的随机种子。比如图像超分任务，LR和HR图像要经过相同的随机剪裁。这种时候可以使用`SyncHandler`:
```python
from rainbowneko.train.data.handler import SyncHandler, ImageHandler

SyncHandler(
    LR=ImageHandler(...),
    HR=ImageHandler(...),
)
```

## 数据流控制

Handler中可以进行数据流的控制，可以指定输入输出数据的流向。通过`key_map_in`和`key_map_out`可以控制输入输出的流向:
```python
ImageHandler(..., , key_map_in=('image -> image',), key_map_out=('image -> image_weak',))
```

```{tip}
如果输入是dict或者list的嵌套，比如`{'data': {'image': img, 'label': label}}`，可以使用`('data.image -> image', 'data.label -> label')
`这种方式来指定输入的流向。

`{'data':[img, label]}`这样的，可以通过索引来指定`('data.0 -> image', 'data.1 -> label')`

输出也可以用类似的操作`('image -> data.image', 'label -> data.label')`
```