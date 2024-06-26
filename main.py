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

from utils import SimCLRDataset
from ssl_model import SimCLR
from utils import contrastive_loss

image_dir = os.getenv('IMAGE_DIR') 
checkpoints_path = os.getenv('CHECKPOINTS_PATH')

print(f'Number of GPUs available: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

os.makedirs('checkpoints', exist_ok=True)

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = SimCLRDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# Using a pre-trained ResNet18 as the base model
backbone = 'resnet18'
base_model = models.resnet18(pretrained=False)
model = SimCLR(base_model, out_dim=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = nn.DataParallel(model)  

print(f'Model is using devices: {model.device_ids}')

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
best_loss = float('inf')

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for (x_i, x_j) in progress_bar:
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        optimizer.zero_grad()
        
        z_i = model(x_i)
        z_j = model(x_j)
        
        loss = contrastive_loss(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/len(dataloader))
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint_path = os.path.join(checkpoints_path, f'best_simclr_model_{backbone}_loss_{best_loss:.4f}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved best model with loss: {best_loss:.4f}')
