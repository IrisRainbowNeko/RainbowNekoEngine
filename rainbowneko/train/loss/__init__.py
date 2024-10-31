from .base import LossContainer, LossGroup
from .distillation import DistillationLoss
from .contrastive import MLCEImageLoss, InfoNCELoss, NoisyInfoNCELoss
from .style import StyleLoss
from .classify import SoftCELoss
from .pseudo_label import PseudoLabelLoss
from .charbonnier import CharbonnierLoss
from .ms_ssim import MSSSIMLoss