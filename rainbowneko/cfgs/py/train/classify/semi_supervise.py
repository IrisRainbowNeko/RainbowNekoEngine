from functools import partial

import torch
import torchvision
import torchvision.transforms as T
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.transforms import InterpolationMode

from cfgs.py.train.classify import multi_class
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.evaluate import MetricGroup, MetricContainer
from rainbowneko.models.ema import ModelEMA
from rainbowneko.models.wrapper import DistillationWrapper
from rainbowneko.parser import CfgModelParser
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data import FixedBucket
from rainbowneko.train.data.handler import HandlerChain, ImageHandler, LoadImageHandler, HandlerGroup
from rainbowneko.train.data.source import IndexSource, UnLabelSource
from rainbowneko.train.loss import LossContainer, PseudoLabelLoss

num_classes = 10


def load_resnet(model, path=None):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if path:
        model.load_state_dict(torch.load(path)['base'])
    return model

def make_cfg():
    dict(
        _base_=[multi_class],

        model_part=CfgModelParser([
            dict(
                lr=1e-2,
                layers=['model_student'],
            )
        ]),

        ckpt_manager=CkptManagerPKL(_partial_=True, saved_model=(
            {'model': 'model_student', 'trainable': False},
        )),

        train=dict(
            train_epochs=100,
            save_step=2000,

            loss=dict(_replace_=True,
                dataset_S=LossContainer(CrossEntropyLoss(), key_map=('pred.pred_student -> 0', 'inputs.label -> 1')),
                dataset_U=LossContainer(PseudoLabelLoss(), key_map=('pred.pred_student -> pred', 'pred.pred_teacher -> pred_label')),
            ),

            metrics=dict(_replace_=True,
                dataset_S=MetricGroup(metric_dict=dict(
                    acc=MetricContainer(MulticlassAccuracy(num_classes=num_classes), key_map=('pred.pred_student -> 0', 'inputs.label -> 1')),
                    f1=MetricContainer(MulticlassF1Score(num_classes=num_classes), key_map=('pred.pred_student -> 0', 'inputs.label -> 1')),
                )),
                dataset_U=None,
            ),
        ),

        model=dict(
            name='cifar-resnet18',
            wrapper=DistillationWrapper(_partial_=True, _replace_=True,
                key_map_in_student=dict(
                    dataset_S=('image -> 0',),
                    dataset_U=('image_strong -> 0',),
                ),
                key_map_in_teacher=dict(
                    dataset_S={},
                    dataset_U=('image_weak -> 0',),
                ),
                key_map_out=('pred_student -> pred_student', 'pred_teacher -> pred_teacher'),
                model_teacher=load_resnet(torchvision.models.resnet18()),
                model_student=load_resnet(torchvision.models.resnet18()),
                ema=ModelEMA
            )
        ),

        data_train=dict(_replace_=True,
            dataset_S=partial(BaseDataset, batch_size=16, loss_weight=1.0,
                source=dict(
                    data_source1=IndexSource(
                        data=torchvision.datasets.cifar.CIFAR10(root=r'D:\others\dataset\cifar', train=True, download=True)
                    ),
                ),
                handler=HandlerChain(handlers=dict(
                    load=LoadImageHandler(),
                    bucket=FixedBucket.handler, # bucket 会自带一些处理模块
                    image=ImageHandler(transform=create_transform(
                            input_size=224,
                            is_training=True,
                            color_jitter=0.4,
                            auto_augment='rand-m9-mstd0.5-inc1',
                            interpolation='bicubic',
                            re_prob=0.25,
                            re_mode='pixel',
                            re_count=1,
                            mean=IMAGENET_DEFAULT_MEAN,
                            std=IMAGENET_DEFAULT_STD,
                        )
                    )
                )),
                bucket=FixedBucket(target_size=224),
            ),
            dataset_U=partial(BaseDataset, batch_size=16, loss_weight=1.0,
                source=dict(
                    data_source1=UnLabelSource(
                        img_root=r'E:\dataset\k40\k40'
                    ),
                ),
                handler=HandlerGroup(handlers=dict(
                    strong=HandlerChain(handlers=dict(
                        load=LoadImageHandler(),
                        image=ImageHandler(transform=create_transform(
                                input_size=224,
                                is_training=True,
                                color_jitter=0.4,
                                auto_augment='rand-m9-mstd0.5-inc1',
                                interpolation='bicubic',
                                re_prob=0.25,
                                re_mode='pixel',
                                re_count=1,
                                mean=IMAGENET_DEFAULT_MEAN,
                                std=IMAGENET_DEFAULT_STD,
                            )
                        )
                    ), key_map_out=('image -> image_strong',)),
                    weak=HandlerChain(handlers=dict(
                        load=LoadImageHandler(),
                        image=ImageHandler(transform=T.Compose([
                            T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                        ])
                        )
                    ), key_map_out=('image -> image_weak',)),
                )),
                bucket=FixedBucket(target_size=224),
            )
        ),
    )
