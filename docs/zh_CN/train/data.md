# 数据集配置

本节会介绍关于数据配置的使用方法，以及数据加载部分大致的组织架构。通过本文档，你将学会如何灵活地使用配置文件来加载、处理、组织和增强数据集。

---

## 概述

在RainbowNeko Engine中，数据集的定义通过配置文件完成。推荐使用**python**配置文件，允许用户通过灵活的方式定义数据源、数据处理逻辑、数据分桶策略等内容。

以下是一个典型的数据集配置示例：

```python
@neko_cfg
def cfg_data():
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

---

## 数据集的核心组件
数据集的配置主要由以下几个核心组件组成：

1. **数据源 (DataSource)**  
可以通过`DataSource`及其子类实现。
   + 定义了数据的来源，例如从本地文件加载、从远程API获取等。
   + 定义了数据的结构，如`(图片,标签)`或`(图片,图片,文本)`这样的数据结构。

2. **数据处理器 (DataHandler)**  
数据处理器用于对原始数据进行处理，例如图像读取、图像增强、数据格式转换等。可以通过`HandlerChain`或`HandlerGroup`组合多个处理器。

3. **数据分桶 (Bucket)**  
Bucket用于对数据进行分组，例如把尺寸相等的图片分到一个batch中。可以通过`BaseBucket`及其子类实现。

4. **数据集 (Dataset)**  
数据集是对数据源、处理器和分桶的封装，提供标准的`__getitem__`和`__len__`接口。

---

## 数据集配置详解

### 1. 数据源配置

数据源定义到`Dataset`的`source`参数中，是一个dict，支持多种数据源类型。以下是一个典型的数据源配置：

```python
source=dict(
    data_source1=IndexSource(
        data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True)
    ),
)
```

:::{dropdown} 数据源类型
:animate: fade-in

- **`IndexSource`**  
直接从一个可以索引的对象(提供`__getitem__`和`__len__`接口)中加载数据，比如pytorch自己的`Dataset`。

- **`ImageLabelSource`**  
  从图像文件夹和标签文件中加载数据。
  ```python
  data_source1=ImageLabelSource(
      img_root='图片文件夹路径',
      label_file='标签文件路径',
  )
  ```
  `label_file`支持的格式见[标注文件格式](./label_file.md)
  
  数据结构:
  ```
  {
    'image': 图片路径,
    'label': 标签
  }
  ```

- **`ImagePairSource`**  
  从图像文件夹和标签文件中加载数据，图像与图像配对的数据。
  ```python
  data_source1=ImagePairSource(
      img_root='图片文件夹路径',
      label_file='标签文件路径',
  )
  ```
  
  数据结构:
  ```
  {
    'image': 图片路径1,
    'label': 图片路径2
  }
  ```
  
- **`ImageFolderClassSource`**  
  每个类别的数据存放在不同文件夹中，适用于分类模型。
  ```
  dataset/
  ├── class1/
  │   ├── img1.png
  │   └── img2.png
  ├── class2/
  │   └── ...
  └── ...
  ```
  使用方法
  ```python
  data_source1=ImageFolderClassSource(
      img_root='数据文件夹路径',
      use_cls_index=True, # True 使用类别id, False 使用类名,
  )
  ```
  
  数据结构:
  ```
  {
    'image': 图片路径,
    'label': 类名 或 类别id
  }
  ```

- **`UnLabelSource`**  
  无标签数据源，只有数据本身。
  ```python
  data_source1=UnLabelSource(
      img_root='数据文件夹路径',
  )
  ```
  
  数据结构:
  ```
  {
    'image': 图片路径,
  }
  ```

:::

#### 多数据源

数据源可以定义多个，会在handler处理后合并到一起，由bucket统一分组。

```python
source=dict(
    data_source1=...,
    data_source2=...,
)
```

---

### 2. 数据处理器配置

数据处理器通过`Dataset`的`handler`字段定义，用于对数据进行预处理或增强。以下是一个常用的图片处理器配置：

```python
handler=HandlerChain(handlers=dict(
    load=LoadImageHandler(), # 读取图像
    bucket=FixedBucket.handler, # Bucket内置处理器
    # 图像变换与增强
    image=ImageHandler(transform=T.Compose([
            T.RandomCrop(size=32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]),
    )
))
```

```{tip}
在常见使用场景中，只需要修改ImageHandler中的transform部分即可。更详细复杂的处理器配置使用方法请参考[数据处理高级配置](../train_adv/handler.md)。
```

#### batch处理器

`BaseDataset`支持添加batch处理器，针对如MixUP等需要以batch为单位进行处理的操作。比如添加MixUP操作，对图片和标签在batch内进行混合：
```python
from rainbowneko.train.data.handler import MixUPHandler

dataset1=BaseDataset(
    batch_handler=HandlerChain(handlers=dict(
        mixup=MixUPHandler(num_classes=num_classes)
    ))
)
```

---

### 3. 数据分桶配置

数据桶通过`bucket`字段定义，用于对数据进行分组，确保一个batch内的图片有相同的尺寸。

+ 如果你的训练任务图像大小都一样，或无所谓剪裁，可以使用`FixedBucket`。它会将所有图像按短边缩放并剪裁到指定大小。
  ```python
  from rainbowneko.train.data import FixedBucket
  
  bucket=FixedBucket(target_size=32) # 32x32
  ```

+ 如果你的训练任务对图片剪裁很敏感，你希望图片尽可能不被剪裁，可以使用`RatioBucket`。它会设置几个不同宽高比的bucket，将图片放入宽高比最接近的bucket，缩放并剪裁至bucket的分辨率，这样可以尽可能减少剪裁。
  ```python
  from rainbowneko.train.data import RatioBucket
  
  # from_files会读取数据集所有图片的分辨率，并根据它们的宽高比进行聚类，找到最合适的分桶方式
  bucket=RatioBucket.from_files(
      target_area=512*512, # 每个桶的预期像素数量
      step_size=8, # 桶的分辨率宽和高要是step_size的倍数
      num_bucket=10, # 桶的数量
  )
  
  # from_ratios会直接均匀的构造桶，不考虑数据集的情况
  bucket=RatioBucket.from_ratios(
      target_area=512*512, # 每个桶的预期像素数量
      step_size=8, # 桶的分辨率宽和高要是step_size的倍数
      num_bucket=10, # 桶的数量
      ratio_max=4, # 最大宽高比，宽高比范围是 [1/ratio_max, ratio_max]
  )
  ```

+ 如果你的任务对图像缩放和剪裁都很敏感，可以使用`SizeBucket`。它会设置几个不同分辨率的bucket，将图片放入分辨率最接近的bucket，剪裁图像至bucket的分辨率。使用这一bucket时，数据集最好数量足够多。
  ```python
  from rainbowneko.train.data import SizeBucket
  
  # from_files会读取数据集所有图片的分辨率，并根据它们的宽和高进行聚类，找到最合适的分桶方式
  bucket=SizeBucket.from_files(
      step_size=8, # 桶的分辨率宽和高要是step_size的倍数
      num_bucket=10, # 桶的数量
  )
  ```


---

### 4. 数据集封装

数据集通过`BaseDataset`类封装数据源、处理器和分桶。配置一些额外的参数。

数据集可以配置多个，每一个有自己的`batch_size`和`loss_weight`。在训练时，不同数据集会独立进行前向与反向传播，并将梯度加起来，因此可以有不同的分辨率或输入格式。`loss_weight`表示这一个数据集在计算loss时的权重。
```python
dict(
    dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
        source=...,
        handler=...,
        bucket=...,
    ),
    dataset2=partial(BaseDataset, batch_size=32, loss_weight=0.2,
        source=...,
        handler=...,
        bucket=...,
    )
)
```

---

## 配置自己的数据集

以下是ImageNet数据集配置示例：

```python
@neko_cfg
def cfg_data():
    dict(
        dataset1=partial(BaseDataset, batch_size=64, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='./imagenet'),
            ),
            handler=HandlerChain(handlers=dict(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(224),
                        T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
                )
            )),
            bucket=FixedBucket(target_size=224),
        )
    )
```

---