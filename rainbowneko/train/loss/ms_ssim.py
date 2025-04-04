import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable

class MSSSIMLoss(torch.nn.Module):
    def __init__(self, size_average=True, max_val=1):
        super(MSSSIMLoss, self).__init__()
        self.size_average = size_average
        self.max_val = max_val
        self.weight = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, sigma, channel):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).contiguous()
        return window

    def _ssim(self, img1, img2, size_average = True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = self.create_window(window_size, sigma, c).to(img1.device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=c) - mu1_mu2

        C1 = (0.01 * self.max_val) **2
        C2 = (0.03 * self.max_val) **2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):
        self.weight = self.weight.to(img1.device)
        msssim = torch.empty(levels, device=img1.device)
        mcs = torch.empty(levels, device=img1.device)
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1] ** self.weight[0:levels-1])*
                                    (msssim[levels-1] ** self.weight[levels-1]))
        return value


    def forward(self, img1, img2):
        img1 = torch.relu(img1)
        img2 = torch.relu(img2)
        return 1-self.ms_ssim(img1, img2)