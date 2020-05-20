import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()

        self.metric = nn.MSELoss()

    def forward(self, pred, gt, mask):
        error = mask * self.metric(pred, gt)
        return error.sum() / (mask > 0).sum().float()


