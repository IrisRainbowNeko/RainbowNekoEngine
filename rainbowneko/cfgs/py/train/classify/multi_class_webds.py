from functools import partial

from cfgs.py.train.classify import multi_class
from rainbowneko.data import WebDataset, WebDSImageLabelSource, HandlerChain, LoadImageHandler, FixedBucket, ImageHandler
from rainbowneko.parser import neko_cfg, CfgWDModelParser
from rainbowneko.data.source import image_pipeline
import torchvision.transforms as T

num_classes = 10
multi_class.num_classes = num_classes

@neko_cfg
def make_cfg():
    return dict(
        _base_=[multi_class],

        model_part=CfgWDModelParser([
            dict(
                lr=1e-4,
                layers=[''],  # train all layers
            )
        ], weight_decay=1e-3),

        data_train=cfg_data(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(WebDataset, batch_size=16, loss_weight=1.0,
            source=dict(
                data_source1=WebDSImageLabelSource(
                    pipeline=image_pipeline('cifar10.tar'),
                    label_file='cifar10_labels.json'
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler, # bucket 会自带一些处理模块
                image=ImageHandler(transform=T.Compose([
                        T.RandomCrop(size=32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ]),
                )
            ),
            bucket=FixedBucket(target_size=32),
        )
    )