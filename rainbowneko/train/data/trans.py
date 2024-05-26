from torch.nn import functional as F
import torch

class ImageTrans:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, data):
        data['img'] = self.trans(data['img'])
        return data

class ImageLabelTrans:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, data):
        img, label = self.trans(data['img'], data['label'])
        data['img'] = img
        data['label'] = label
        return data

class MixUP:
    def __init__(self, num_classes: int, alpha: float = 1.0, ratio: float = 1.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.ratio = ratio
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def __call__(self, data):
        img, label = data['img'], data['label']['label']

        if label.dim() == 1:
            label = F.one_hot(label, num_classes=self.num_classes)

        label = label.float()
        N_mix = int(img.shape[0]*self.ratio)
        lam = self._dist.sample((N_mix,))
        lam_i = lam.view(-1, *[1]*(img.dim()-1))
        lam_t = lam.view(-1, 1)

        img[:N_mix] = img[:N_mix].roll(1, 0).mul_(1.0 - lam_i).add_(img[:N_mix].mul(lam_i))
        label[:N_mix] = label[:N_mix].roll(1, 0).mul_(1.0 - lam_t).add_(label[:N_mix].mul(lam_t))

        data['img'] = img
        data['label']['label'] = label
        return data