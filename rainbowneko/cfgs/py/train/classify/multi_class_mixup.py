from cfgs.py.train.classify import multi_class
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data.handler import MixUPHandler, HandlerChain
from rainbowneko.train.loss import LossContainer, SoftCELoss

num_classes = 10
multi_class.num_classes = num_classes


def make_cfg():
    dict(
        _base_=[multi_class],

        train=dict(
            loss=LossContainer(loss=SoftCELoss()),
            metrics=None,
        ),

        data_train=dict(
            dataset1=BaseDataset(
                batch_handler=HandlerChain(handlers=dict(
                    mixup=MixUPHandler(num_classes=num_classes)
                ))
            )
        ),
    )
