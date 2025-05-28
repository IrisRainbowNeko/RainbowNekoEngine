import torch
import torchvision
import torchvision.transforms as T
from torch import nn

from rainbowneko.data import IndexSource, HandlerChain, LoadImageHandler, ImageHandler, BaseDataset, BaseBucket
from rainbowneko.infer import HandlerAction, DataLoaderAction
from rainbowneko.infer.workflow import (Actions, BuildModelAction, PrepareAction, FeedAction, ForwardAction,
                                        LambdaAction, VisClassAction, LoadModelAction)
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import auto_ckpt_loader

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@neko_cfg
def infer_one(path):
    return Actions([
        FeedAction(image=path),
        HandlerAction(handler=HandlerChain(
            load=LoadImageHandler(),
            image=ImageHandler(transform=T.Compose([
                T.CenterCrop(size=32),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]))
        ), key_map_in=('image -> image',)),
        LambdaAction(f_act=lambda image, **kwargs: {'image': image.unsqueeze(0)}),
        ForwardAction(key_map_in=('image -> input.image', 'model -> model')),
        VisClassAction(
            class_map=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'],
            key_map_in=('output.pred -> pred',)
        )
    ])

@neko_cfg
def infer_all(path):
    return DataLoaderAction(
        dataset=BaseDataset(_partial_=True, batch_size=1, loss_weight=1.0,
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
            VisClassAction(
                class_map=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'],
                key_map_in=('output.pred -> pred',)
            )
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
        infer_one(path=r"E:\dataset\frog10.png")
        # infer_all(path=r'D:\others\dataset\cifar')
    ]))