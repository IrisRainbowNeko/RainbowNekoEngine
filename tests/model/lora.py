from unittest import TestCase
from rainbowneko.models.lora.lora_layers_patch import LoraLayer

class LoraTester(TestCase):
    def test_lora(self):
        from timm.models.swin_transformer_v2 import swinv2_base_window8_256

        model = swinv2_base_window8_256()
        lora_layers = LoraLayer.wrap_model(0, model)
        print(model)
        print(lora_layers)