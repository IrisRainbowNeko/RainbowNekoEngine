from cfgs.py.train.classify import multi_class
from rainbowneko.data import BaseDataset
from rainbowneko.data.handler import MixUPHandler, HandlerChain
from rainbowneko.parser import neko_cfg
from rainbowneko.train.loss import LossContainer, SoftCELoss

num_classes = 10
multi_class.num_classes = num_classes

@neko_cfg
def make_cfg():
    return dict(
        _base_=[multi_class],

        train=dict(
            loss=LossContainer(loss=SoftCELoss()),
            metrics=None,
        ),

        data_train=dict(
            dataset1=BaseDataset(
                batch_handler=HandlerChain(
                    mixup=MixUPHandler(num_classes=num_classes)
                )
            )
        ),
    )
