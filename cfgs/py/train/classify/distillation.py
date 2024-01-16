from functools import partial

import torch
from torch import nn
import torchvision
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from rainbowneko.evaluate import EvaluatorGroup, ClsEvaluatorContainer
from rainbowneko.models.wrapper import DistillationWrapper
from rainbowneko.train.data import FixedBucket
from rainbowneko.train.data.source import IndexSource
from rainbowneko.train.loss import LossContainer, LossGroup, DistillationLoss
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.ckpt_manager import CkptManagerPKL

num_classes = 10

def load_resnet(model, path=None):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if path:
        model.load_state_dict(torch.load(path)['base'])
    return model

config = dict(
    _base_=[
        'cfgs/py/train/classify/multi_class.py',
    ],

    model_part=[
        dict(
            lr=1e-2,
            layers=['model_student'],
        )
    ],

    ckpt_manager=partial(CkptManagerPKL, saved_model=(
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
                        model_teacher=load_resnet(torchvision.models.resnet50(), r'E:\codes\python_project\RainbowNekoEngine\exps\resnet50-2024-01-16-17-08-57\ckpts\cifar-resnet50-6000.ckpt'),
                        model_student=load_resnet(torchvision.models.resnet18())
                        )
    ),
)
