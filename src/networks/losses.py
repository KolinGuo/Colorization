"""Loss functions"""

from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

def get_loss_func(loss_func: str, class_weight) -> nn.modules.loss._Loss:
    """Returns an instance of the requested loss function"""
    if loss_func == 'MSELoss':
        return MSELoss(reduction='sum')
    raise ValueError('Unknown loss_func')

class MSELoss(nn.MSELoss):
    """Custom MSELoss with 1/2 factor"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(size_average=size_average, 
                                      reduce=reduce, 
                                      reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction) / 2.0 / input.shape[0]

class MSELoss_Vibrant(nn.MSELoss):
    """Custom MSELoss with 1/2 factor"""
    pass
