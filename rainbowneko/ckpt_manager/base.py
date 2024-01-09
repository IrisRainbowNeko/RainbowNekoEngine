from diffusers import StableDiffusionPipeline
from diffusers.models.lora import LoRACompatibleLinear

class CkptManagerBase:
    def __init__(self, **kwargs):
        pass

    def set_save_dir(self, save_dir, emb_dir=None):
        raise NotImplementedError()

    def save(self, name, step, model, all_plugin, ema={}, **kwargs):
        raise NotImplementedError()

    @classmethod
    def load(cls, pretrained_model, **kwargs) -> StableDiffusionPipeline:
        raise NotImplementedError
