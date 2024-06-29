from functools import partial

import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from cfgs.py.train import train_base, tuning_base
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.evaluate import MetricGroup, MetricContainer, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import make_base, CfgModelParser
from rainbowneko.train.data import FixedBucket
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.train.data.source import IndexSource
from rainbowneko.train.loss import LossContainer

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_cfg():
    dict(
        _base_=make_base(train_base, tuning_base)+[],

        model_part=CfgModelParser([
            dict(
                lr=1e-2,
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

            loss=partial(LossContainer, loss=CrossEntropyLoss()),

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

        data_train=dict(
            dataset1=partial(ImageLabelDataset, batch_size=128, loss_weight=1.0,
                source=dict(
                    data_source1=IndexSource(
                        data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True),
                        image_transforms=torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(size=32, padding=4),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                        ])
                    ),
                ),
                bucket=FixedBucket(target_size=32),
            )
        ),

        evaluator=partial(Evaluator,
            interval=500,
            metric=MetricGroup(metric_dict=dict(
                acc=MetricContainer(MulticlassAccuracy(num_classes=num_classes)),
                f1=MetricContainer(MulticlassF1Score(num_classes=num_classes)),
            )),
            dataset=dict(
                dataset1=partial(ImageLabelDataset, batch_size=128, loss_weight=1.0,
                    source=dict(
                        data_source1=IndexSource(
                            data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=False, download=True),
                            image_transforms=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                            ])
                        ),
                    ),
                    bucket=FixedBucket(target_size=32),
                )
            )
        ),
    )
