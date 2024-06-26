import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv

load_dotenv()

class CNNModel(nn.Module):
    def __init__(self, encoder):
        super(CNNModel, self).__init__()
        self.encoder = encoder
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # Adjust upsample factor
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.segmentation_head(x)
        return x
