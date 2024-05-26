from .base import LossContainer, LossGroup
from .distillation import DistillationLoss
from .contrastive import MLCEImageLoss, InfoNCELoss, NoisyInfoNCELoss
from .style import StyleLoss
from .classify import SoftCELoss