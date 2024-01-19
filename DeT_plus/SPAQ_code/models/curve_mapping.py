import torch
from torch import nn
import numpy as np
import math


class CurveMapping(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = nn.Parameter(torch.tensor(5.0))
        self.x0 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        k = self.k
        x0 = self.x0
        a = (
            (1 + torch.exp(k * x0))
            * (1 + torch.exp(k * x0 - k))
            / (torch.exp(k * x0) - torch.exp(k * x0 - k))
        )

        b = a / (1 + torch.exp(k * x0))
        y = a / (1 + torch.exp(-k * (x - x0))) - b
        para_list = [
            k.data.cpu().numpy(),
            x0.data.cpu().numpy(),
            a.data.cpu().numpy(),
            b.data.cpu().numpy(),
        ]
        return y, para_list
