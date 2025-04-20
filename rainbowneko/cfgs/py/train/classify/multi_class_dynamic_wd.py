from cfgs.py.train.classify import multi_class
from rainbowneko.parser import neko_cfg, CfgWDModelParser
from rainbowneko.utils.scheduler import CosineWD

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

        train=dict(
            wd_scheduler=CosineWD(
                _partial_=True,
                warmup_steps=10,
                min_scale=10,
            ),
        ),
    )
