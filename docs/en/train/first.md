# Train Your First Model

The RainbowNeko Engine uses configuration files to set up various parameters and modules for training. In this example, we will use a configuration file for a multi-class classification task to train a simple `resnet` model on the `CIFAR-10` dataset.

RainbowNeko Engine supports configuration files in both `python` and `yaml` formats. Here, we will use the `python` format as an example. Configuration files in `python` are written entirely in Python syntax, offering greater flexibility for defining models and data. They are also more readable, making them clearer and easier to use.

## Initialize the Project
```{attention}
RainbowNeko Engine should be used as a library rather than directly running training or inference within its folder.
```

Create a new folder:
```bash
mkdir my_first_project
cd my_first_project
```

Initialize the project, which will automatically create the necessary configuration files in the current folder:
```bash
nekoinit
```

## Start Training

::::{tab-set}
:::{tab-item} Single-GPU Training
Run the following command, specifying the configuration file, to start training your model. All data and model definitions are included in the configuration file.
```bash
# Train with Accelerate and only one GPU
neko_train_1gpu --cfg cfgs/py/train/classify/multi_class.py
```
:::

:::{tab-item} Multi-GPU Training
For multi-GPU training, specify the GPU IDs and number of GPUs to use in `cfgs/launcher/multi.yaml`, then run:
```bash
# Train with Accelerate and multiple GPUs
neko_train --cfg cfgs/py/train/classify/multi_class.py
```

````{tip}
You can also copy the `cfgs/launcher/multi.yaml` configuration file, modify its parameters, and specify the new configuration file using the `--launch_cfg` parameter.
>   ```bash
>   neko_train --launch_cfg cfgs/launcher/multi_2.yaml --cfg cfgs/py/train/classify/multi_class.py
>   ```
````
:::
::::

## Adjust Training Parameters
This section explains how to adjust training-related parameters in the configuration file.

```{note}
In configuration files, settings are defined within the `make_cfg` function using combinations of `dict`, `list`, and various class or function calls.
```

### Adjusting Learning Rate
In the configuration file, `model_part` defines training parameters for each layer of the model. Adjusting the `lr` parameter changes the learning rate:
```python
model_part=CfgWDModelParser([
    dict(
        lr=2e-4,  # Set learning rate to 2e-4
        layers=[''],  # Train all layers
    )
]),
```

```{tip}
Using `model_part`, you can set individual learning rates and other parameters for each layer. For more details, see [Layered Training](../train_adv/layer_train.md).
```

### Adjusting Other Parameters
Under the `train` section, you can define commonly used training parameters:
```python
train=dict(
    train_steps=1000,  # Total training steps
    train_epochs=100,  # Total epochs; if not None, this overrides train_steps
    workers=2,  # Number of processes for data loading
    max_grad_norm=None,  # Gradient clipping threshold
    save_step=2000,  # Interval (in steps) for saving models
    gradient_accumulation_steps=1,  # Gradient accumulation steps
)
```

## Modify Model Configuration

In the configuration file's `model` section, you define which model to use for training:
```python
def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model=dict(
    name='cifar-resnet18',  # Model name used when saving checkpoints 
    wrapper=partial(SingleWrapper, model=load_resnet())  # Model structure defined by load_resnet()
),
```

```{note}
Models in RainbowNeko Engine need to be wrapped using a Wrapper. For tasks with only one model and a single data flow pipeline, you can use `SingleWrapper`.
```

## Adjust Batch Size

The `batch size` is defined in the data configuration section under `data_train`. The data configurations are placed inside a separate function called `cfg_data()`. Modify the `batch_size` value within this function to adjust it.

```{tip}
Functions decorated with `@neko_cfg` become configuration functions. During parsing of configuration files, their contents are directly placed where they are called.
```

## Change Data Path

In the data configuration section under `dataset`, the `source` parameter defines where your dataset comes from. To change your dataset's location, modify the CIFAR10's root path inside `data_source1`.

```{note}
For more details about dataset configurations, refer to [Dataset Configuration](./data.md).
```