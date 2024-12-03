from rainbowneko.infer.workflow import (Actions, BuildModelAction, PrepareAction, LoopAction, FeedAction, ForwardAction,
                                        VisPredAction, LambdaAction, VisClassAction, LoadModelAction)
from rainbowneko.models.wrapper import SingleWrapper
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from rainbowneko.train.data import IndexSource, HandlerChain, LoadImageHandler, ImageHandler
from rainbowneko.infer import HandlerAction
from rainbowneko.utils import neko_cfg
from rainbowneko.ckpt_manager import CkptManagerPKL

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@neko_cfg
def infer_one(path):
    Actions([
        FeedAction(image=path),
        HandlerAction(handler=HandlerChain(handlers=dict(
            load=LoadImageHandler(),
            image=ImageHandler(transform=T.Compose([
                T.CenterCrop(size=32),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]))
        )), key_map_in=('image -> image',)),
        LambdaAction(f_act=lambda image, **kwargs: {'image': image.unsqueeze(0)}),
        ForwardAction(key_map_in=('image -> input.image', 'model -> model')),
        VisClassAction(
            class_map=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'],
            key_map_in=('output.pred -> pred',)
        )
    ])

@neko_cfg
def infer_all(path):
    LoopAction(
        iterator=IndexSource(
            _partial_=True,
            data=torchvision.datasets.cifar.CIFAR10(root=path, train=False, download=True)
        ),
        actions=[
            HandlerAction(handler=HandlerChain(handlers=dict(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                    T.CenterCrop(size=32),
                    T.ToTensor(),
                    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]))
            ))),
            LambdaAction(f_act=lambda image, **kwargs: {'image': image.unsqueeze(0)}),
            ForwardAction(key_map_in=('image -> input.image', 'model -> model')),
            VisClassAction(
                class_map=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'],
                key_map_in=('output.pred -> pred',)
            )
        ]
    )

def make_cfg():
    Actions(actions=[
        PrepareAction(device='cpu', dtype=torch.float16),
        BuildModelAction(SingleWrapper(_partial_=True, model=load_resnet())),
        LoadModelAction(dict(
            part=[dict(
                module_to_load='model',
                path='exps/2024-12-02-14-19-39/ckpts/cifar-resnet18-3900.ckpt',
                alpha=1.0
            )]
        )),
        infer_one(path=r"E:\dataset\frog10.png")
        #infer_all(path=r'D:\others\dataset\cifar')
    ])