import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import DenseNet
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
import matplotlib.pyplot as plt

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

model = CombinedSSLModel().cuda()
checkpoint_path = '/data/aravind/dfu_ssl/checkpoints_fine_tune/dino_attn_unet_best.pth'
checkpoint = torch.load(checkpoint_path)
if 'state_dict' not in checkpoint:
    state_dict = checkpoint
else:
    state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')
    if new_key in model.state_dict():
        new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_mask(image, output_size=(640, 480)):
    image_tensor = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output).cpu().numpy().squeeze()
        mask = output > 0.5
        mask = mask.astype(np.uint8) * 255
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.resize(output_size, Image.NEAREST)
        return mask_image

def process_images(input_folder, output_folder, output_size=(640, 480)):
    os.makedirs(output_folder, exist_ok=True)
    for image_name in os.listdir(input_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_name)
            image = Image.open(image_path).convert('RGB')
            predicted_mask = predict_mask(image, output_size=output_size)
            output_filename = os.path.splitext(image_name)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            predicted_mask.save(output_path)
            print(f'Saved predicted mask for {image_name} at {output_path}')

input_folder = '/data/aravind/DFU2024/DFUC2024_test_release'
output_folder = '/data/aravind/dfu_ssl/outputs'
process_images(input_folder, output_folder, output_size=(640, 480))
