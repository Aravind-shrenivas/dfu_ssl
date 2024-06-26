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

from utils import DFUMaskDataset
from seg_model import CNNModel
from utils import remove_module_prefix

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0).float())  # Ensure mask is binary (0 or 1)
])

csv_file = os.getenv('DFU2022_CSV_PATH')
image_dir = os.getenv('DFU2022_IMAGE_DIR')  
mask_dir = os.getenv('DFU2022_MASK_DIR')
checkpoints_path = os.getenv('CHECKPOINTS_PATH') 
checkpoints_fine_tune = os.getenv('CHECKPOINTS_FINE_TUNE')   

labelled_dataset = DFUMaskDataset(csv_file, image_dir, mask_dir, transform, mask_transform)
labelled_dataloader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

# Using a pre-trained ResNet18 as the base model
base_model = models.resnet18(pretrained=False)
base_model = nn.Sequential(*list(base_model.children())[:-2])
segmentation_model = CNNModel(base_model)

# Load pre-trained weights
checkpoint_path = os.path.join(checkpoints_path,'best_simclr_model_resnet18_loss_4.0592.pth')  # Change this to your checkpoint path
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Remove 'module.' prefix from state dict keys
state_dict = remove_module_prefix(state_dict)

encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder.')}
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items()}
segmentation_model.encoder.load_state_dict(encoder_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model = segmentation_model.to(device)
segmentation_model = nn.DataParallel(segmentation_model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=0.001)

# Fine-tuning loop
num_epochs = 10
best_loss = float('inf')

segmentation_model.train()
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(labelled_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = segmentation_model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/len(labelled_dataloader))

    avg_loss = total_loss / len(labelled_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint_path = os.path.join(checkpoints_fine_tune, f'fine_tuned_model_loss_{best_loss:.4f}.pth')
        torch.save(segmentation_model.state_dict(), checkpoint_path)
        print(f'Saved best model with loss: {best_loss:.4f}')
