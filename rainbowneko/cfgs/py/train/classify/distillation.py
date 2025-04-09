from functools import partial

import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss

from cfgs.py.train.classify import multi_class
from rainbowneko.ckpt_manager import ckpt_saver
from rainbowneko.models.wrapper import DistillationWrapper
from rainbowneko.parser import CfgModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer, LossGroup, DistillationLoss

num_classes = 10


def load_resnet(model, path=None):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if path:
        model.load_state_dict(torch.load(path)['base'])
    return model

@neko_cfg
def make_cfg():
    return dict(
        _base_=[multi_class],

        model_part=CfgModelParser([
            dict(
                lr=1e-2,
                layers=['model_student'],
            )
        ]),

        ckpt_saver=dict(
            model=ckpt_saver(target_module='model_student')
        ),

        train=dict(
            train_epochs=100,
            save_step=2000,

            loss=LossGroup(_replace_=True, loss_list=[
                LossContainer(CrossEntropyLoss(), weight=0.05),
                LossContainer(DistillationLoss(T=5.0, weight=0.95)),
            ]),
        ),

        model=dict(
            name='cifar-resnet18',
            wrapper=partial(DistillationWrapper, _replace_=True,
                            model_teacher=load_resnet(torchvision.models.resnet50()),
                            model_student=load_resnet(torchvision.models.resnet18())
                            )
        ),
    )
