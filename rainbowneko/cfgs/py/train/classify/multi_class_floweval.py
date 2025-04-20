from functools import partial

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from cfgs.py.eval import multi_class_flow
from cfgs.py.train import train_base, tuning_base
from rainbowneko.ckpt_manager import ckpt_saver
from rainbowneko.data import BaseDataset
from rainbowneko.data import FixedBucket
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.data.source import IndexSource
from rainbowneko.evaluate import MetricGroup, MetricContainer, WorkflowEvaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer
from rainbowneko.utils import CosineLR

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],

        model_part=CfgWDModelParser([
            dict(
                lr=1e-4,
                layers=[''],  # train all layers
            )
        ]),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_saver=dict(
            model=ckpt_saver(target_module='model')
        ),

        train=dict(
            train_epochs=10,
            workers=2,
            max_grad_norm=None,
            save_step=2000,

            loss=LossContainer(loss=CrossEntropyLoss()),

            optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=100,
            ),
            metrics=MetricGroup(
                acc=MetricContainer(MulticlassAccuracy(num_classes=num_classes, average='micro')),
                f1=MetricContainer(MulticlassF1Score(num_classes=num_classes, average='micro')),
            ),
        ),

        model=dict(
            name='cifar-resnet18',
            wrapper=partial(SingleWrapper, model=load_resnet())
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=WorkflowEvaluator(_partial_=True,
            interval=50,
            workflow=multi_class_flow,
        ),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=16, loss_weight=1.0,
            source=dict(
                data_source1=IndexSource(
                    data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True)
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler, # bucket 会自带一些处理模块
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