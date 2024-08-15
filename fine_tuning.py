import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from torchvision.models import DenseNet
from transformers import AutoModel
import numpy as np
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        if Wg.size()[2:] != Ws.size()[2:]:
            Ws = F.interpolate(Ws, size=Wg.shape[2:], mode='bilinear', align_corners=True)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = conv_block(in_c[0]+out_c, out_c)
    def forward(self, x, s):
        x = self.up(x)
        if x.size() != s.size():
            s = F.interpolate(s, size=x.shape[2:], mode='bilinear', align_corners=True)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(768, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.b1 = conv_block(256, 512)
        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.upsample_output = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        b1 = self.b1(p3)
        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        output = self.output(d3)
        output = self.upsample_output(output)
        return output

class MultiScaleDenseNet(nn.Module):
    def __init__(self, in_channels, out_channels=128, output_size=(14, 14)):
        super(MultiScaleDenseNet, self).__init__()
        self.densenet3x3 = DenseNet(
            growth_rate=16, block_config=(6, 6, 6, 6), num_init_features=64, 
            bn_size=4, drop_rate=0, num_classes=out_channels
        )
        self.densenet5x5 = DenseNet(
            growth_rate=16, block_config=(6, 6, 6, 6), num_init_features=64, 
            bn_size=4, drop_rate=0, num_classes=out_channels
        )
        self.densenet7x7 = DenseNet(
            growth_rate=16, block_config=(6, 6, 6, 6), num_init_features=64, 
            bn_size=4, drop_rate=0, num_classes=out_channels
        )
        self.reduce_conv3x3 = nn.Conv2d(188, 128, kernel_size=1)
        self.reduce_conv5x5 = nn.Conv2d(188, 128, kernel_size=1)
        self.reduce_conv7x7 = nn.Conv2d(188, 128, kernel_size=1)
        self.reduce_channels = nn.Conv2d(384, 384, kernel_size=1)
        self.resize_output = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
    def forward(self, x):
        out3x3 = self.densenet3x3.features(x)
        out5x5 = self.densenet5x5.features(x)
        out7x7 = self.densenet7x7.features(x)
        out3x3 = self.reduce_conv3x3(out3x3)
        out5x5 = self.reduce_conv5x5(out5x5)
        out7x7 = self.reduce_conv7x7(out7x7)
        out = torch.cat([out3x3, out5x5, out7x7], dim=1)
        out = self.reduce_channels(out)
        out = self.resize_output(out)
        return out

class CombinedSSLModel(nn.Module):
    def __init__(self, in_channels=3, multi_scale_channels=128, target_dim=20):
        super(CombinedSSLModel, self).__init__()
        self.vit_model = AutoModel.from_pretrained('facebook/dinov2-large')
        self.multi_scale_extractor = MultiScaleDenseNet(in_channels=in_channels, out_channels=multi_scale_channels, output_size=(target_dim, target_dim))
        self.fusion_layer = nn.Conv2d(1408, 768, kernel_size=1)
        self.resize_vit_features = nn.Upsample(size=(target_dim, target_dim), mode='bilinear', align_corners=False)
        self.attention_unet = attention_unet()
    def forward(self, x):
        vit_outputs = self.vit_model(pixel_values=x)
        vit_features = vit_outputs.last_hidden_state
        batch_size, num_patches, hidden_size = vit_features.shape
        spatial_dim = int(np.sqrt(num_patches))
        vit_features = vit_features[:, 1:, :].transpose(1, 2).reshape(batch_size, hidden_size, spatial_dim, spatial_dim)
        vit_features = self.resize_vit_features(vit_features)
        multi_scale_features = self.multi_scale_extractor(x)
        combined_features = torch.cat([vit_features, multi_scale_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        mask_output = self.attention_unet(fused_features)
        return mask_output

class SegmentationDataset(Dataset):
    def __init__(self, csv_file, images_folder, masks_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx, 0]
        mask_name = self.df.iloc[idx, 1]
        image_path = os.path.join(self.images_folder, image_name)
        mask_path = os.path.join(self.masks_folder, mask_name)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            mask = self.transform(mask)
        return image, mask

def dice_coefficient(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean()

def dice_loss(pred, target, epsilon=1e-6):
    return 1 - dice_coefficient(pred, target, epsilon)

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

csv_file = '/data/aravind/DFU2022/matching_files.csv'
images_folder = '/data/aravind/DFU2022/images'
masks_folder = '/data/aravind/DFU2022/mask'
model_weights_path = '/data/aravind/dfu_ssl/checkpoints/dino_cnn_ssl_bs64.pth'

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dataset = SegmentationDataset(csv_file, images_folder, masks_folder, transform=transform)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = CombinedSSLModel().cuda()

checkpoint = torch.load(model_weights_path)

if 'state_dict' not in checkpoint:
    state_dict = checkpoint
else:
    state_dict = checkpoint['state_dict']

new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')
    if new_key in model.state_dict():
        new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)

for param in model.vit_model.parameters():
    param.requires_grad = False

for param in model.multi_scale_extractor.parameters():
    param.requires_grad = False

model = nn.DataParallel(model)

optimizer = optim.Adam([
    {'params': model.module.vit_model.parameters(), 'lr': 1e-5},
    {'params': model.module.multi_scale_extractor.parameters(), 'lr': 1e-5},
    {'params': model.module.attention_unet.parameters(), 'lr': 1e-4},
])

criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

num_epochs = 20
freeze_epochs = 5
best_val_loss = float('inf')

for epoch in range(num_epochs):
    if epoch == freeze_epochs:
        for param in model.module.vit_model.parameters():
            param.requires_grad = True
        for param in model.module.multi_scale_extractor.parameters():
            param.requires_grad = True
    
    model.train()
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update()
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.cuda()
            masks = masks.cuda()
            outputs = model(images)
            val_loss += dice_loss(outputs, masks).item()
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), '/data/aravind/dfu_ssl/checkpoints_fine_tune/dino_attn_unet_best.pth')
        print(f"Best model saved at Epoch {epoch+1} with Validation Loss: {best_val_loss:.4f}")
