from cfgs.py.train.classify import multi_class
from rainbowneko.ckpt_manager import ckpt_saver, LocalCkptSource, NekoModelModuleSaver
from rainbowneko.ckpt_manager.format import ONNXFormat
from rainbowneko.parser import neko_cfg


@neko_cfg
def make_cfg():
    return dict(
        _base_=[multi_class],

        ckpt_saver=dict(
            model=ckpt_saver(target_module='model'),
            model_onnx=NekoModelModuleSaver(
                format=ONNXFormat(inputs={'image': (('batch', 1), 3, 32, 32)}),
                source=LocalCkptSource(),
                target_module='model'
            )
        ),

    )
