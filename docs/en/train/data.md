# Dataset Configuration

This section introduces how to use dataset configuration and provides an overview of the dataset architecture. Through this document, you will learn how to flexibly use configuration files to load, process, organize, and augment datasets.

---

## Overview

In the RainbowNeko Engine, datasets are defined through configuration files. It is recommended to use **Python** configuration files, which allow users to define data sources, data processing logic, data bucketing strategies, and more in a flexible manner.

Below is a typical example of a dataset configuration:

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

---

## Core Components of a Dataset
The dataset configuration primarily consists of the following core components:

1. **Data Source (DataSource)**  
   Implemented via `DataSource` and its subclasses.
   + Defines the source of the data (e.g., loading from local files or fetching from remote APIs).
   + Specifies the structure of the data (e.g., `(image, label)` or `(image, image, text)`).

2. **Data Handler (DataHandler)**  
   Used for processing raw data such as image reading, augmentation, and format conversion. Multiple handlers can be combined using `HandlerChain` or `HandlerGroup`.

3. **Data Bucket (Bucket)**  
   Organizes data into groups (e.g., grouping images with the same size into one batch). Implemented via `BaseBucket` and its subclasses.

4. **Dataset**  
   Wraps the data source, handler, and bucket while providing standard `__getitem__` and `__len__` interfaces.

---

## Detailed Dataset Configuration

### 1. Data Source Configuration

The data source is defined in the `source` parameter of `Dataset`. It is a dictionary that supports various types of data sources. Below is an example of a typical data source configuration:

```python
source=dict(
    data_source1=IndexSource(
        data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True)
    ),
)
```

:::{dropdown} Data Source Types
:animate: fade-in

- **`IndexSource`**  
  Loads data directly from an indexable object (providing `__getitem__` and `__len__` interfaces), such as PyTorch's built-in `Dataset`.

- **`ImageLabelSource`**  
  Loads images and labels from an image folder and a label file.
  ```python
  data_source1=ImageLabelSource(
      img_root='Path to image folder',
      label_file='Path to label file',
  )
  ```
  Supported formats for `label_file` can be found in [Label File Formats](./label_file.md).
  
  Data structure:
  ```
  {
    'image': Path to image,
    'label': Label
  }
  ```

- **`ImagePairSource`**  
  Loads paired image-to-image datasets from an image folder and a label file.
  ```python
  data_source1=ImagePairSource(
      img_root='Path to image folder',
      label_file='Path to label file',
  )
  ```
  
  Data structure:
  ```
  {
    'image': Path to first image,
    'label': Path to second image
  }
  ```

- **`ImageFolderClassSource`**  
  Stores each class's images in separate folders; suitable for classification models.
  ```
  dataset/
  ├── class1/
  │   ├── img1.png
  │   └── img2.png
  ├── class2/
      └── ...
      ...
  ```
  
  Usage:
  ```python
  data_source1=ImageFolderClassSource(
      img_root='Path to dataset folder',
      use_cls_index=True, # True for using class IDs; False for using class names.
  )
  ```
  
  Data structure:
  ```
  {
    'image': Path to image,
    'label': Class name or class ID
  }
  ```

- **`UnLabelSource`**  
   Unlabeled datasets containing only raw images.
   ```python
   data_source1=UnLabelSource(
       img_root='Path to dataset folder',
   )
   ```
   
   Data structure:
   ```
   {
     'image': Path to image,
   }
   ```

:::

#### Multiple Data Sources

You can define multiple data sources that will be merged after being processed by handlers and grouped by buckets.

```python
source=dict(
    data_source1=...,
    data_source2=...,
)
```

---

### 2. Data Handler Configuration

Data handlers are defined in the `handler` field of `Dataset`, used for preprocessing or augmenting the data. Below is an example of a commonly used image handler configuration:

```python
handler=HandlerChain(
    load=LoadImageHandler(), # Reads images.
    bucket=FixedBucket.handler, # Built-in bucket handler.
    # Image transformation and augmentation.
    image=ImageHandler(transform=T.Compose([
            T.RandomCrop(size=32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]),
    )
)
```

```{tip}
In most common scenarios, you only need to modify the transform section within ImageHandler. For more advanced handler configurations, refer to [Advanced Data Processing Configurations](../train_adv/handler.md).
```

#### Batch Handlers

The `BaseDataset` supports adding batch handlers for operations like MixUP that require processing at the batch level. For instance:

```python
from rainbowneko.data.handler import MixUPHandler

dataset1 = BaseDataset(
    batch_handler=HandlerChain(
        mixup=MixUPHandler(num_classes=num_classes)
    )
)
```

---

### 3. Data Bucket Configuration

Buckets are defined via the `bucket` field and are used for grouping datasets so that all images in a batch have consistent sizes.

+ If all your training images have identical sizes or cropping is not critical: use `FixedBucket`. It scales all images by their shorter side and crops them to a fixed size.

```python
from rainbowneko.data import FixedBucket

bucket = FixedBucket(target_size=32)  # Resizes to dimensions of size (32x32).
```

+ For tasks sensitive to cropping but not scaling: use `RatioBucket`. It clusters images into buckets based on aspect ratios while minimizing cropping.

```python
from rainbowneko.data import RatioBucket

bucket = RatioBucket.from_files(
    target_area=512 * 512,
    step_size=8,
    num_bucket=10,
)
```

+ For tasks sensitive to both scaling and cropping: use `SizeBucket`. It clusters based on resolution similarity instead of aspect ratio.

```python
from rainbowneko.data import SizeBucket

bucket = SizeBucket.from_files(step_size=8, num_bucket=10, )
```

---

### Dataset Wrapping

Datasets are wrapped using the `BaseDataset` class by combining sources with handlers/buckets alongside additional parameters.

Multiple datasets can be configured simultaneously with independent resolutions or input formats during training.

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

In this configuration:
- `batch_size` specifies the number of samples in each batch for the respective dataset.
- `loss_weight` determines the weight of this dataset's loss when calculating the total loss during training.

---

## Configuring Your Own Dataset

Below is an example configuration for the ImageNet dataset:

```python
@neko_cfg
def cfg_data():
    dict(
        dataset1=partial(BaseDataset, batch_size=64, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='./imagenet'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(224),
                        T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
                )
            ),
            bucket=FixedBucket(target_size=224),
        )
    )
```

### Explanation of the Configuration:
1. **Data Source**:  
   The `ImageFolderClassSource` is used to load ImageNet data where images are organized by class folders.

2. **Handlers**:  
   - `LoadImageHandler`: Reads image files.
   - `ImageHandler`: Applies a series of transformations and augmentations:
     - Resize images to a fixed size of 224x224.
     - Randomly crop a resized region to ensure variation.
     - Apply horizontal flipping with a certain probability.
     - Normalize pixel values using ImageNet's mean and standard deviation.

3. **Bucket**:  
   A `FixedBucket` ensures that all images are resized to a uniform size (224x224) for efficient batching.

4. **Batch Size and Loss Weight**:  
   The dataset is configured with a batch size of 64 and a loss weight of 1.0, meaning its contribution to the total loss is fully weighted.

---

By following this guide, you can configure datasets tailored to your specific needs in the RainbowNeko Engine framework. For further customization or advanced features such as integrating custom handlers or buckets, refer to the advanced documentation sections linked within this guide.
