# Configuring Loss Functions

## Single Loss Function
A loss function needs to be wrapped in a `LossContainer`. For example, to use cross-entropy loss:
```python
from rainbowneko.train.loss import LossContainer

loss = LossContainer(loss=CrossEntropyLoss())
```

```{note}
It is recommended that the loss function placed inside `LossContainer` inherits from `nn.Module`.
```

## Multiple Loss Functions
You can combine multiple loss functions using `LossGroup`, and assign weights to each loss:
```python
from rainbowneko.train.loss import LossContainer, LossGroup

loss = LossGroup([
    LossContainer(loss=CrossEntropyLoss()),
    LossContainer(loss=MSELoss(), weight=0.2),
])
```

## Data Flow Control

With the `key_map` parameter of `LossContainer`, you can specify which variables are used to compute the loss. For example, in a semi-supervised learning scenario:
```python
LossContainer(CrossEntropyLoss(), key_map=('pred.pred_student -> 0', 'inputs.label -> 1'))
```
Here, `pred` represents the model's output, and `inputs` represents all input data. The prediction output `pred_student` from the model is mapped as the 0th input for the loss, while the `label` from the input data is mapped as the 1st input.

By default, the `key_map` of `LossContainer` is set to `('pred.pred -> 0', 'inputs.label -> 1')`. This means that under default settings, the 0th input for the loss function is the model's output, and the 1st input is the `label` from the input data.