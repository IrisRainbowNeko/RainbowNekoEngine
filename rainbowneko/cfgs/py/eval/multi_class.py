import time

import torchvision
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision import transforms as T

from rainbowneko.ckpt_manager import NekoResumer, auto_ckpt_loader
from rainbowneko.data import BaseDataset, ImageHandler, IndexSource, HandlerChain, LoadImageHandler, FixedBucket
from rainbowneko.evaluate import Evaluator, MetricGroup, MetricContainer
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import neko_cfg
from rainbowneko.loggers import CLILogger

time_format="%Y-%m-%d-%H-%M-%S"
num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@neko_cfg
def make_cfg():
    return Evaluator(
        _partial_=True,
        exp_dir=f'exps_eval/{time.strftime(time_format)}',
        mixed_precision='fp16',
        seed=42,

        interval=50,
        metric=MetricGroup(
            acc=MetricContainer(MulticlassAccuracy(num_classes=num_classes, average='micro')),
            f1=MetricContainer(MulticlassF1Score(num_classes=num_classes, average='micro')),
        ),
        model=dict(
            name='cifar-resnet18',
            enable_xformers=True,
            gradient_checkpointing=False,
            force_cast_precision=False,

            wrapper=SingleWrapper(_partial_=True, model=load_resnet())
        ),
        resume=NekoResumer(
            loader=dict(
                model=auto_ckpt_loader(
                    target_module='model',
                    path='exps/cifar/ckpts/cifar-resnet18-3900.ckpt',
                ),
            )
        ),
        logger=[
            CLILogger(_partial_=True, out_path='eval.log', log_step=20),
        ],
        dataset=BaseDataset(_partial_=True, batch_size=128, workers=4, loss_weight=1.0,
            source=dict(
                data_source1=IndexSource(
                    data=torchvision.datasets.cifar.CIFAR10(root='/mnt/others/dataset/cifar', train=False, download=True)
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler,
                image=ImageHandler(transform=T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ]),
                )
            ),
            bucket=FixedBucket(target_size=32),
        )
    )