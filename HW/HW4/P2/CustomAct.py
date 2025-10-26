import torch
import numpy as np

# def custom_activation(x):
#     # Triangle wave
#     return 2 * torch.abs(2 * (x - torch.floor(x + 0.5))) - 1


def triangle_wave(x):
    period = 2 * np.pi  # Same period as sin(x)
    scaled_x = x / period + 1/4
    return 2 * torch.abs(2 * (scaled_x - torch.floor(scaled_x + 0.5))) - 1

def custom_activation(x, sharpness=10):
    return torch.tanh(sharpness * torch.sin(x))