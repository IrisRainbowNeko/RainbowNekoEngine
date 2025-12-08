from torch import nn
from rainbowneko.utils import KeyMapper

class LossContainer(nn.Module):
    def __init__(self, loss, weight=1.0, key_map=None):
        super().__init__()
        self.key_mapper = KeyMapper(loss, key_map)
        self.loss = loss
        self.weight = weight

    def forward(self, pred, inputs):
        args, kwargs = self.key_mapper(pred=pred, inputs=inputs)
        if isinstance(self.loss, FullInputLoss):
            return self.loss(*args, _full_pred=pred, _full_inputs=inputs, **kwargs) * self.weight
        else:
            return self.loss(*args, **kwargs) * self.weight

class LossGroup(nn.Module):
    def __init__(self, *loss_list, **loss_dict):
        """
        支持两种配置方式：
        1) LossGroup([LossContainer(...), LossContainer(...)])
        2) LossGroup(mse=LossContainer(...), gw=LossContainer(...), ...)
        3) LossGroup(loss_list=[LossContainer(...), ...])
        """
        super().__init__()

        # 兼容 LossGroup([loss1, loss2, ...])：把单个 list/tuple 展开
        if len(loss_list) == 1 and isinstance(loss_list[0], (list, tuple)):
            loss_list = tuple(loss_list[0])

        # 未命名的 loss，自动生成名称 loss_0, loss_1, ...
        unnamed_losses = {f"loss_{idx}": loss for idx, loss in enumerate(loss_list)}

        # 兼容 LossGroup(loss_list=[...]) 这种写法
        if "loss_list" in loss_dict and isinstance(loss_dict["loss_list"], (list, tuple)):
            offset = len(unnamed_losses)
            extra_losses = {
                f"loss_{offset + idx}": loss
                for idx, loss in enumerate(loss_dict.pop("loss_list"))
            }
            unnamed_losses.update(extra_losses)

        # 过滤掉类似 _replace_ 这类控制字段，只保留真正的 loss
        named_losses = {
            name: loss
            for name, loss in loss_dict.items()
            if not (isinstance(name, str) and name.startswith("_") and name.endswith("_"))
        }

        all_losses = {**unnamed_losses, **named_losses}
        # 注册为 ModuleDict，方便 to(device)、state_dict 等管理
        self.loss_dict = nn.ModuleDict(all_losses)

    def forward(self, pred, target):
        loss = {}
        for name, loss_item in self.loss_dict.items():
            loss[name] = loss_item(pred, target).squeeze()
        return loss

class FullInputLoss:
    def forward(self, *args, _full_pred, _full_inputs, **kwargs):
        return self.loss(*args, **kwargs)