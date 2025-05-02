import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from rainbowneko.data import IndexSource, HandlerChain, LoadImageHandler, ImageHandler, BaseDataset, BaseBucket
from rainbowneko.evaluate import MetricGroup, MetricContainer
from rainbowneko.infer import DataLoaderAction, MetricAction
from rainbowneko.infer.workflow import (Actions, BuildModelAction, PrepareAction, ForwardAction,
                                        LoadModelAction)
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import auto_ckpt_loader

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@neko_cfg
def infer_all(path):
    return DataLoaderAction(
        dataset=BaseDataset(_partial_=True, batch_size=32, loss_weight=1.0,
            source=dict(
                data_source1=IndexSource(
                    data=torchvision.datasets.cifar.CIFAR10(root=path, train=False, download=True)
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.CenterCrop(size=32),
                        T.ToTensor(),
                        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ]),
                )
            ),
            bucket=BaseBucket(),
        ),
        actions=Actions([
            ForwardAction(key_map_in=('image -> input.image', 'model -> model', 'device -> device', 'dtype -> dtype')),
            MetricAction(metric=MetricGroup(
                acc=MetricContainer(MulticlassAccuracy(num_classes=num_classes, average='micro')),
                f1=MetricContainer(MulticlassF1Score(num_classes=num_classes, average='micro')),
            ), key_map_in=('output -> pred', 'label -> inputs.label', 'device -> device'))
        ])
    )

@neko_cfg
def make_cfg():
    return dict(workflow=Actions(actions=[
        PrepareAction(device='cpu', dtype=torch.float16),
        BuildModelAction(SingleWrapper(_partial_=True, model=load_resnet())),
        LoadModelAction(dict(
            model=auto_ckpt_loader(
                target_module='model',
                path='exps/cifar/ckpts/cifar-resnet18-3900.ckpt',
            ),
        )),
        #infer_one(path=r"E:\dataset\frog10.png")
        infer_all(path=r'D:\others\dataset\cifar')
    ]))