from functools import partial

import torch
from torch import nn
import torchvision
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from rainbowneko.evaluate import EvaluatorGroup, ClsEvaluatorContainer
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.train.data import FixedBucket
from rainbowneko.train.data.source import IndexSource
from rainbowneko.train.loss import LossContainer
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.ckpt_manager import CkptManagerPKL

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

config = dict(
    _base_=[
        'cfgs/py/train/train_base.py',
        'cfgs/py/train/tuning_base.py',
    ],

    model_part=[
        dict(
            lr=1e-2,
            layers=[''],  # train all layers
        )
    ],

    ckpt_manager=partial(CkptManagerPKL, saved_model=(
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
        metrics=partial(EvaluatorGroup, evaluator_dict=dict(
            acc=ClsEvaluatorContainer(MulticlassAccuracy(num_classes=num_classes)),
            f1=ClsEvaluatorContainer(MulticlassF1Score(num_classes=num_classes)),
        )),
    ),

    model=dict(
        name='cifar-resnet18',
        wrapper=partial(SingleWrapper, model=load_resnet())
    ),

    evaluator=partial(EvaluatorGroup,
        interval=500,
        evaluator_dict=dict(
            acc=ClsEvaluatorContainer(MulticlassAccuracy(num_classes=num_classes)),
            f1=ClsEvaluatorContainer(MulticlassF1Score(num_classes=num_classes)),
        )
    ),

    data_train=dict(
        dataset1=partial(ImageLabelDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=IndexSource(
                    data=torchvision.datasets.cifar.CIFAR10(root=r'E:\dataset\cifar', train=True, download=True),
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

    data_eval=dict(
        dataset1=partial(ImageLabelDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=IndexSource(
                    data=torchvision.datasets.cifar.CIFAR10(root=r'E:\dataset\cifar', train=False, download=True),
                    image_transforms=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ])
                ),
            ),
            bucket=FixedBucket(target_size=32),
        )
    )
)
