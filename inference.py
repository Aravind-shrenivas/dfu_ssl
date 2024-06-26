import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
from dotenv import load_dotenv
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

load_dotenv()

from seg_model import CNNModel
from utils import remove_module_prefix
from utils import visualize_and_save

checkpoints_fine_tune = os.getenv('CHECKPOINTS_FINE_TUNE')

# ensure these match the training transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Using a pre-trained ResNet18 as the base model
base_model = models.resnet18(pretrained=False)
base_model = nn.Sequential(*list(base_model.children())[:-2])
segmentation_model = CNNModel(base_model)

finetuned_checkpoint_path = os.path.join(checkpoints_fine_tune,'fine_tuned_model_loss_0.0425.pth')  # Change this to your fine-tuned weights path
state_dict = torch.load(finetuned_checkpoint_path, map_location=torch.device('cpu'))

# Remove 'module.' prefix from state dict keys
state_dict = remove_module_prefix(state_dict)
segmentation_model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model = segmentation_model.to(device)
segmentation_model = nn.DataParallel(segmentation_model)

segmentation_model.eval()

test_image_dir = os.getenv('TEST_IMAGE_DIR')
output_dir = os.getenv('OUTPUT_DIR')  
os.makedirs(output_dir, exist_ok=True)

for test_image_name in tqdm(os.listdir(test_image_dir), desc='Processing test images'):
    test_image_path = os.path.join(test_image_dir, test_image_name)
    output_path = os.path.join(output_dir, f'{os.path.splitext(test_image_name)[0]}_output.png')

    test_image = transform(Image.open(test_image_path).convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_mask = segmentation_model(test_image)
        predicted_mask = torch.sigmoid(predicted_mask)  
        predicted_mask = (predicted_mask > 0.5).float()  

    visualize_and_save(test_image_path, predicted_mask, output_path)

print(f'Inference complete. Outputs saved to {output_dir}')

