from functools import partial

import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss

from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.models.wrapper import DistillationWrapper
from rainbowneko.train.loss import LossContainer, LossGroup, DistillationLoss
from rainbowneko.parser import make_base, CfgModelParser

from cfgs.py.train.classify import multi_class

num_classes = 10


def load_resnet(model, path=None):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if path:
        model.load_state_dict(torch.load(path)['base'])
    return model

def make_cfg():
    dict(
        _base_=make_base(multi_class)+[],

        model_part=CfgModelParser([
            dict(
                lr=1e-2,
                layers=['model_student'],
            )
        ]),

        ckpt_manager=partial(CkptManagerPKL, _partial_=True, saved_model=(
            {'model': 'model_student', 'trainable': False},
        )),

        train=dict(
            train_epochs=100,
            save_step=2000,

            loss=partial(LossGroup, loss_list=[
                LossContainer(CrossEntropyLoss(), weight=0.05),
                DistillationLoss(T=5.0, weight=0.95),
            ]),
        ),

        model=dict(
            name='cifar-resnet18',
            wrapper=partial(DistillationWrapper,
                            model_teacher=load_resnet(torchvision.models.resnet50(),
                                                      r'E:\codes\python_project\RainbowNekoEngine\exps\resnet50-2024-01-16-17-08-57\ckpts\cifar-resnet50-6000.ckpt'),
                            model_student=load_resnet(torchvision.models.resnet18())
                            )
        ),
    )
