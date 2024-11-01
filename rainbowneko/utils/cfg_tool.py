from torch import nn
from typing import Callable


def change_num_classes(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def neko_cfg(f_cfg: Callable):
    f_cfg._neko_cfg_ = True
    return f_cfg

def load_resnet(num_classes=10):
    import torchvision
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model