import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
import random