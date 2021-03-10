"""Loss functions"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

def get_loss_func(loss_func: str, class_weight,
                  color_vivid_gamma: float = 2.0) -> nn.modules.loss._Loss:
    """Returns an instance of the requested loss function"""
    if loss_func == 'MSELoss':
        return MSELoss(reduction='sum')
    if loss_func == 'MSELoss_Vibrant':
        return MSELoss_Vibrant(reduction='sum',
                               color_vivid_gamma=color_vivid_gamma)
    raise ValueError('Unknown loss_func')

class MSELoss(nn.MSELoss):
    """Custom MSELoss with 1/2 factor"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(size_average=size_average, 
                                      reduce=reduce, 
                                      reduction=reduction)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.mse_loss(y_pred, y_true, reduction=self.reduction) / 2.0 / y_pred.shape[0]  # batch_size

class MSELoss_Vibrant(nn.MSELoss):
    """Custom MSELoss_Vibrant with 1/2 factor"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 color_vivid_gamma: float = 2.0) -> None:
        super(MSELoss_Vibrant, self).__init__(size_average=size_average,
                                              reduce=reduce,
                                              reduction=reduction)
        self.color_vivid_gamma = color_vivid_gamma

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        c_pred = torch.norm(y_pred, dim=1)
        c_true = torch.norm(y_true, dim=1)  # []
        ab_loss = F.mse_loss(y_pred, y_true, reduction=self.reduction) / 2.0
        chroma_loss = 2 * F.mse_loss(c_pred, c_true, reduction=self.reduction)

        return (ab_loss + self.color_vivid_gamma * chroma_loss) / y_pred.shape[0]  # batch_size
