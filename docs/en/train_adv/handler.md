# Advanced Data Processing Configuration

## Processing Chain

The `HandlerChain` allows you to sequentially combine multiple data processing modules. The output of each module overwrites the input content and is passed as the input to the next module.

```{image} ../../imgs/handler_chain.svg
:alt: Select Parameters
:width: 500px
:align: center
```

For example, common image processing:
```python
from rainbowneko.train.data.handler import HandlerChain, ImageHandler, LoadImageHandler

handler = HandlerChain(handlers=dict(
    load=LoadImageHandler(),
    bucket=FixedBucket.handler,  # The bucket includes some built-in processing modules
    image=ImageHandler(transform=T.Compose([
            T.RandomCrop(size=32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]),
    )
)),
```

## Processing Group

The `HandlerChain` also allows for parallel combination of multiple data processing modules. Data is input into each module, and their outputs are aggregated to produce the final output.

```{image} ../../imgs/handler_group.svg
:alt: Select Parameters
:width: 400px
:align: center
```

For instance, reading an image and applying different processing steps to store results in separate variables:
```python
from rainbowneko.train.data.handler import HandlerGroup, HandlerChain, ImageHandler, LoadImageHandler

handler = HandlerChain(handlers=dict(
    load=LoadImageHandler(),
    image=HandlerGroup(handlers=dict(
        weak=ImageHandler(..., key_map_out=('image -> image_weak',)),
        strong=ImageHandler(..., key_map_out=('image -> image_strong',)),
    ))
)),
```

## Synchronizing Random Seeds for Processing

In certain scenarios, multiple processing modules may need to use the same random seed. For example, in super-resolution tasks where both low-resolution (LR) and high-resolution (HR) images require identical random cropping. In such cases, you can use `SyncHandler`:

```python
from rainbowneko.train.data.handler import SyncHandler, ImageHandler

SyncHandler(handlers=dict(
    LR=ImageHandler(...),
    HR=ImageHandler(...),
))
```

## Data Flow Control

Handlers allow control over data flow by specifying the direction of input and output data using `key_map_in` and `key_map_out`:
```python
ImageHandler(..., key_map_in=('image -> image',), key_map_out=('image -> image_weak',))
```

```{tip}
If the input is a nested structure like a dictionary or list (e.g., `{'data': {'image': img, 'label': label}}`), you can specify the input flow using syntax like `('data.image -> image', 'data.label -> label')`.

For structures like `{'data': [img, label]}`, you can use indexing to specify inputs as follows: `('data.0 -> image', 'data.1 -> label')`.

Similarly, outputs can be mapped using operations like `('image -> data.image', 'label -> data.label')`.
```