import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

class SimCLRDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
        return image1, image2

class DFUMaskDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, transform=None, mask_transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        mask_path = os.path.join(self.mask_dir, self.data.iloc[idx, 1])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    
    similarity_matrix = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    labels = torch.arange(batch_size).repeat(2).to(z.device)
    
    # Create mask to exclude diagonal elements
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    
    # Select positives
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)], dim=0)
    
    # Select negatives
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
    
    # Ensure the shapes match
    positives = positives.view(2 * batch_size, 1)
    logits = torch.cat((positives, negatives), dim=1)
    
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)
    
    logits = logits / temperature
    loss = nn.functional.cross_entropy(logits, labels)
    
    return loss

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def visualize_and_save(image_path, mask, output_path):
    original_image = Image.open(image_path).convert("RGB")
    mask = mask.squeeze().cpu().detach().numpy()
    
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_resized.resize(original_image.size, Image.NEAREST)
    mask_resized = np.array(mask_resized) / 255.0

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image)

    plt.subplot(1, 3, 2)
    plt.title('Predicted Mask')
    plt.imshow(mask_resized, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    overlay = original_image.copy()
    overlay = np.array(overlay)
    overlay[mask_resized > 0.5, :] = [255, 0, 0]  # Red color for mask
    plt.imshow(overlay)

    plt.savefig(output_path)
    plt.close()