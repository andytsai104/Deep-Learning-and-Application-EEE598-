import torch.nn as nn
import torch
class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, torch.pow(x, 2), 0.2 * x)