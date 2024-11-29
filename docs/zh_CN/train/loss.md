# 配置loss函数

## 单个loss函数
loss需要被包装在`LossContainer`中，比如交叉熵损失：
```python
from rainbowneko.train.loss import LossContainer

loss=LossContainer(loss=CrossEntropyLoss())
```

```{note}
放在`LossContainer`中的loos最好继承自`nn.Module`
```

## 多个loss函数
通过`LossGroup`可以组合多个loss函数，并为每个loss设置权重：
```python
from rainbowneko.train.loss import LossContainer, LossGroup

loss=LossGroup([
    LossContainer(loss=CrossEntropyLoss()),
    LossContainer(loss=MSELoss(), weight=0.2),
])
```

## 数据流控制

我们通过`LossContainer`的`key_map`可以指定用哪些变量去计算loss，比如对于半监督学习场景:
```python
LossContainer(CrossEntropyLoss(), key_map=('pred.pred_student -> 0', 'inputs.label -> 1'))
```
其中`pred`是模型输出结果，`inputs`是输入的所有数据。
把模型预测输出中的`pred_student`作为loss的第0个输入，把输入数据中的`label`作为第1个输入。

`LossContainer`默认的`key_map`是`('pred.pred -> 0', 'inputs.label -> 1')`，即默认情况下，loss的第0个输入是模型的输出，第1个输入是输入数据中的`label`。
