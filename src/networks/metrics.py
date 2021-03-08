"""Custom metrics based on ignite.metrics.Metric"""
# pylint: disable=too-many-ancestors
from typing import Callable, Tuple, Union

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import EpochMetric as EpochMetric

def auc_compute_fn(pos_cnt: torch.Tensor, tot_cnt: int) -> float:
    """Compute AUC score"""
    # pos_cnt: [self.thresholds.shape]
    return torch.sum(pos_cnt / tot_cnt) / torch.numel(pos_cnt)

class AUC(EpochMetric):
    """Calculate Raw Accuracy (AuC) based on paper 3.1 part 3"""
    def __init__(self,
                 step_size: float = 1.0,
                 output_transform: Callable = lambda x: x,
                 check_compute_fn: bool = False,
                 device: Union[str, torch.device] = torch.device("cpu")) -> None:

        self.step_size = step_size
        self.thresholds = torch.arange(0, 150+step_size, step=step_size).to(device)

        super(AUC, self).__init__(auc_compute_fn,
                                  output_transform=output_transform,
                                  check_compute_fn=check_compute_fn,
                                  device=device)

    def reset(self) -> None:
        self.pos_cnt = torch.zeros_like(self.thresholds).to(self._device)
        self.tot_cnt = 0

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        y_pred = y_pred.clone().to(self._device)
        y = y.clone().to(self._device)

        # Compute the distance between y_pred and y
        dist = torch.norm(y_pred-y, dim=1)  # [batch_size, 256, 256]

        # Increment positive pixel counts
        self.pos_cnt += torch.sum(
                torch.le(dist[...,None], self.thresholds[None,None,None,...]),
                dim=(0,1,2))
        # Increment total pixel counts
        self.tot_cnt += torch.numel(dist)

    def compute(self) -> float:
        if self.tot_cnt == 0:
            raise NotComputableError("EpochMetric must have at least one example before it can be computed.")

        return self.compute_fn(self.pos_cnt, self.tot_cnt)


