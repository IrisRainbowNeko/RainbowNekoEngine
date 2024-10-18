from .base import DataHandler

class LabelHandler(DataHandler):
    def __init__(self, transform=None, key_map_in=('label -> label',), key_map_out=('label -> label',)):
        super().__init__(key_map_in, key_map_out)
        self.transform = transform

    def handle(self, label):
        label = self.transform(label)
        return {'label': label}
