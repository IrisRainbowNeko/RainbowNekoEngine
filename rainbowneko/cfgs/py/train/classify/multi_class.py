from functools import partial

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from cfgs.py.train import train_base, tuning_base
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.evaluate import MetricGroup, MetricContainer, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import make_base, CfgWDModelParser
from rainbowneko.train.data import FixedBucket
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data.source import IndexSource
from rainbowneko.train.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.train.loss import LossContainer
from rainbowneko.utils import neko_cfg

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_cfg():
    dict(
        _base_=make_base(train_base, tuning_base)+[],

        model_part=CfgWDModelParser([
            dict(
                lr=1e-4,
                layers=[''],  # train all layers
            )
        ]),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_manager=CkptManagerPKL(_partial_=True, saved_model=(
            {'model':'model', 'trainable':False},
        )),

        train=dict(
            train_epochs=100,
            workers=2,
            max_grad_norm=None,
            save_step=2000,

            loss=LossContainer(loss=CrossEntropyLoss()),

            optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

            scale_lr=False,
            scheduler=dict(
                name='cosine',
                num_warmup_steps=10,
            ),
            metrics=MetricGroup(metric_dict=dict(
                acc=MetricContainer(MulticlassAccuracy(num_classes=num_classes)),
                f1=MetricContainer(MulticlassF1Score(num_classes=num_classes)),
            )),
        ),

        model=dict(
            name='cifar-resnet18',
            wrapper=partial(SingleWrapper, model=load_resnet())
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    dict(
        dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=IndexSource(
                    data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True)
                ),
            ),
            handler=HandlerChain(handlers=dict(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler, # bucket 会自带一些处理模块
                image=ImageHandler(transform=T.Compose([
                        T.RandomCrop(size=32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ]),
                )
            )),
            bucket=FixedBucket(target_size=32),
        )
    )

@neko_cfg
def cfg_evaluator():
    partial(Evaluator,
        interval=500,
        metric=MetricGroup(metric_dict=dict(
            acc=MetricContainer(MulticlassAccuracy(num_classes=num_classes)),
            f1=MetricContainer(MulticlassF1Score(num_classes=num_classes)),
        )),
        dataset=dict(
            dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
                source=dict(
                    data_source1=IndexSource(
                        data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=False, download=True)
                    ),
                ),
                handler=HandlerChain(handlers=dict(
                    load=LoadImageHandler(),
                    bucket=FixedBucket.handler,
                    image=ImageHandler(transform=T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                        ]),
                    )
                )),
                bucket=FixedBucket(target_size=32),
            )
        )
    )