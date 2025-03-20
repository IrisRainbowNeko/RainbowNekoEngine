from .loader import NekoLoader

class NekoResumer:
    def __init__(self, loader, start_step, loader_ema=None):
        self.loader = loader
        self.start_step = start_step
        self.loader_ema = loader_ema

    def load_to(self, model, ema_model=None):
        NekoLoader.load_all(model, self.loader)
        if ema_model is not None:
            NekoLoader.load_all(ema_model, self.loader_ema)
