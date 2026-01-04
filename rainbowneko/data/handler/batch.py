import random

import torch
from torch.nn import functional as F

from .base import DataHandler


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


class CutMixHandler(DataHandler):
    def __init__(self, num_classes: int, alpha: float = 1.0, ratio: float = 1.0,
                 key_map_in=('image -> image', 'label -> label'),
                 key_map_out=('image -> image', 'label -> label')):
        super().__init__(key_map_in, key_map_out)
        self.num_classes = num_classes
        self.alpha = alpha
        self.ratio = ratio
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _rand_bbox(self, H, W, lam):
        """Generate a random bounding box based on the CutMix paper."""
        cut_ratio = torch.sqrt(1.0 - lam)  # cut size ratio
        cut_h = (H * cut_ratio).int()
        cut_w = (W * cut_ratio).int()

        # Random center point
        cy = random.randint(0, H - 1)
        cx = random.randint(0, W - 1)

        # Bounding box boundaries
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)
        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)

        return y1, y2, x1, x2

    def handle(self, image, label):
        """
        image: (N, C, H, W)
        label: (N,) or (N, num_classes)
        """
        if label.dim() == 1:
            label = F.one_hot(label, num_classes=self.num_classes)

        label = label.float()
        N, C, H, W = image.shape
        N_mix = int(N * self.ratio)

        if N_mix <= 0:
            return {'image': image, 'label': label}

        lam = self._dist.sample((N_mix,))  # (N_mix,)
        lam_bbox = torch.zeros(N_mix)

        # Perform CutMix
        index = torch.arange(N_mix).roll(1, 0)  # Align via rolling, similar to MixUpHandler

        for i in range(N_mix):
            # Sample cut region
            y1, y2, x1, x2 = self._rand_bbox(H, W, lam[i])

            # Replace image region
            image[i, :, y1:y2, x1:x2] = image[index[i], :, y1:y2, x1:x2]

            # Calculate actual lambda (since bbox might be clipped at edges)
            area = (y2 - y1) * (x2 - x1)
            lam_bbox[i] = 1.0 - area / (H * W)

        # Calculate label
        lam_bbox = lam_bbox.view(-1, 1)  # (N_mix, 1)
        label[:N_mix] = label[:N_mix] * lam_bbox + label[index] * (1.0 - lam_bbox)

        return {'image': image, 'label': label}
