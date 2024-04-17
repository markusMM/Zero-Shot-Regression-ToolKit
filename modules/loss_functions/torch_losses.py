import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_log_error


class RMSLELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


def RMSLE(y_true, y_pred):  # noqa
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


setattr(nn, 'RMSLELoss', RMSLELoss)


def parse_loss(loss_name: str):
    assert 'loss' in loss_name.lower()
    loss_fcn = getattr(nn, nn.MSELoss)
    return loss_fcn
