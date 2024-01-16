from typing import Union, Dict, Any


class CkptManagerBase:
    def __init__(self, **kwargs):
        pass

    def set_save_dir(self, save_dir):
        raise NotImplementedError()

    def save(self, name, step, model, all_plugin, ema={}, **kwargs):
        raise NotImplementedError()

    @classmethod
    def load(cls, pretrained_model, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_to_model(cls, model, ckpt_path: Union[str, Dict[str, Any]], **kwargs):
        raise NotImplementedError
