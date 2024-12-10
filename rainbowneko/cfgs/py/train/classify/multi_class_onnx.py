from cfgs.py.train.classify import multi_class
from rainbowneko.ckpt_manager import ckpt_manager, LocalCkptSource, ModelManager
from rainbowneko.ckpt_manager.format import ONNXFormat


def make_cfg():
    dict(
        _base_=[multi_class],

        ckpt_manager=[
            ckpt_manager(saved_model=({'model': 'model', 'trainable': False},)),
            ModelManager(
                format=ONNXFormat(inputs={'image': (('batch',1), 3, 32, 32)}),
                source=LocalCkptSource(),
                saved_model=({'model': 'model', 'trainable': False},)
            )
        ],

    )
