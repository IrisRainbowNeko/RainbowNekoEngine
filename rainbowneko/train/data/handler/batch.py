from .base import DataHandler
import torch
from torch.nn import functional as F

class MixUPHandler(DataHandler):
    def __init__(self, num_classes: int, alpha: float = 1.0, ratio: float = 1.0,
                 key_map_in=('image -> image', 'label -> label'), key_map_out=('image -> image', 'label -> label')):
        super().__init__(key_map_in, key_map_out)
        self.num_classes = num_classes
        self.alpha = alpha
        self.ratio = ratio
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def handle(self, image, label):
        if label.dim() == 1:
            label = F.one_hot(label, num_classes=self.num_classes)

        label = label.float()
        N_mix = int(image.shape[0] * self.ratio)
        lam = self._dist.sample((N_mix,))
        lam_i = lam.view(-1, *[1] * (image.dim() - 1))
        lam_t = lam.view(-1, 1)

        image[:N_mix] = image[:N_mix].roll(1, 0).mul_(1.0 - lam_i).add_(image[:N_mix].mul(lam_i))
        label[:N_mix] = label[:N_mix].roll(1, 0).mul_(1.0 - lam_t).add_(label[:N_mix].mul(lam_t))

        return {'image': image, 'label': label}