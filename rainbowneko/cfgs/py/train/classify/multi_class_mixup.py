from functools import partial

from rainbowneko.train.data.trans import MixUP

from cfgs.py.train.classify import multi_class
from rainbowneko.parser import make_base
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.train.data.trans import ImageLabelTrans
from rainbowneko.train.loss import LossContainer, SoftCELoss

num_classes = 10
multi_class.num_classes = num_classes


def make_cfg():
    dict(
        _base_=make_base(multi_class) + [],

        train=dict(
            loss=partial(LossContainer, loss=SoftCELoss()),
            metrics=None,
        ),

        data_train=dict(
            dataset1=ImageLabelDataset(
                batch_transform=MixUP(num_classes=num_classes)
            )
        ),
    )
