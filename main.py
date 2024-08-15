import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torchvision.models import DenseNet

original_mean = [0.6262, 0.5736, 0.5512]
original_std = [0.1557, 0.1816, 0.1933]

augment_transform = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
])

image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

class FootUlcerDataset(Dataset):
    def __init__(self, root, processor, transform=None):
        self.root = root
        self.processor = processor
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith('.jpg') or fname.endswith('.png')]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        processed_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        processed_image_pil = transforms.ToPILImage()(processed_image)
        if self.transform:
            image1 = self.transform(processed_image_pil)
            image2 = self.transform(processed_image_pil)
        else:
            image1 = processed_image_pil
            image2 = processed_image_pil
        image1_tensor = transforms.ToTensor()(image1)
        image2_tensor = transforms.ToTensor()(image2)
        return image1_tensor, image2_tensor, image_path

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
        return fused_features

class WarmUpCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps, last_epoch=-1):
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        super(WarmUpCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            lr = self.warmup_learning_rate + slope * step
        else:
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(step - self.warmup_steps) * torch.pi / (self.total_steps - self.warmup_steps)))
            lr = self.learning_rate_base * cosine_decay
        return [lr for _ in self.base_lrs]

original_folder = '/data/aravind/DFU2024/unlabelled_data_with_val'
validation_folder = '/data/aravind/DFU2024/DFUC2024_test_release'
pseudo_label_save_directory = '/data/aravind/dfu_ssl/pseudo_labels/'
save_directory = '/data/aravind/dfu_ssl/checkpoints/'

batch_sizes = [64]

num_epochs = 40
learning_rate = 1e-5

for batch_size in batch_sizes:
    original_dataloader = DataLoader(FootUlcerDataset(original_folder, processor=image_processor, transform=augment_transform), batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(FootUlcerDataset(validation_folder, processor=image_processor, transform=augment_transform), batch_size=batch_size, shuffle=False)
    model = CombinedSSLModel().cuda()
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    STEPS_PER_EPOCH = len(original_dataloader)
    TOTAL_STEPS = num_epochs * STEPS_PER_EPOCH
    WARMUP_EPOCHS = int(num_epochs * 0.1)
    WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)
    scheduler = WarmUpCosine(
        optimizer=optimizer,
        learning_rate_base=learning_rate,
        total_steps=TOTAL_STEPS,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS
    )
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    save_path = os.path.join(save_directory, f'dino_cnn_ssl_bs{batch_size}.pth')
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(original_dataloader), desc=f'Epoch {epoch+1}/{num_epochs} [Batch Size: {batch_size}]', unit='batch') as pbar:
            for images1, images2, _ in original_dataloader:
                images1 = images1.cuda()
                images2 = images2.cuda()
                features1 = model(images1)
                features2 = model(images2)
                features1 = features1.view(features1.size(0), -1)
                features2 = features2.view(features2.size(0), -1)
                loss = criterion(features1, features2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update()
        current_lr = scheduler.get_lr()[0]
        print(f'Epoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr:.6f}')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images1, images2, _ in validation_dataloader:
                images1 = images1.cuda()
                images2 = images2.cuda()
                features1 = model(images1)
                features2 = model(images2)
                features1 = features1.view(features1.size(0), -1)
                features2 = features2.view(features2.size(0), -1)
                loss = criterion(features1, features2)
                val_loss += loss.item()
        val_loss /= len(validation_dataloader)
        print(f'Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model to {save_path}')
    print(f'Training complete for batch size {batch_size}. Best model saved to {save_path} with validation loss {best_val_loss:.4f}')
