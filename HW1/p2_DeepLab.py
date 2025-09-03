import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models.segmentation as models
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 7

# Use DeepLabV3+ with ResNet50 backbone
def create_model(num_classes, pretrained=True):
    model = models.deeplabv3_resnet50(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(512, 512), augment=True, oversample_factor=2):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith('_sat.jpg')]
        self.oversample_factor = oversample_factor
        
        # Identify images containing Rangeland
        self.rangeland_images = []
        for img_name in self.image_filenames:
            mask_name = img_name.replace('_sat.jpg', '_mask.png')
            mask_path = os.path.join(self.root_dir, mask_name)
            mask = Image.open(mask_path).convert("RGB")
            mask = np.array(mask)
            if np.any((mask == [128, 0, 128]).all(axis=2)):  # Purple color for Rangeland
                self.rangeland_images.append(img_name)
        
        # Oversample Rangeland images
        self.image_filenames.extend(self.rangeland_images * (self.oversample_factor - 1))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.augment:
            image, mask = self.transform(image, mask, img_name in self.rangeland_images)
        else:
            image = TF.resize(image, self.image_size)
            mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask = TF.to_tensor(mask)

        mask = self.rgb_to_class(mask)

        return image, mask

    def transform(self, image, mask, is_rangeland):
        # Resize
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.image_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (only multiples of 90 degrees to preserve pixel alignment)
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Color jitter (only for image)
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2) if is_rangeland else random.uniform(0.9, 1.1)
            contrast = random.uniform(0.8, 1.2) if is_rangeland else random.uniform(0.9, 1.1)
            saturation = random.uniform(0.8, 1.2) if is_rangeland else random.uniform(0.9, 1.1)
            
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)

        # Random Gaussian noise (only for image)
        if is_rangeland and random.random() > 0.7:
            image_np = np.array(image)
            noise = np.random.normal(0, 5, image_np.shape).astype(np.uint8)
            image_np = np.clip(image_np + noise, 0, 255)
            image = Image.fromarray(image_np)

        # Slight random shifts (only for Rangeland)
        if is_rangeland and random.random() > 0.7:
            shift_range = int(min(self.image_size) * 0.1)
            dx = random.randint(-shift_range, shift_range)
            dy = random.randint(-shift_range, shift_range)
            image = ImageOps.expand(image, (shift_range, shift_range, shift_range, shift_range))
            mask = ImageOps.expand(mask, (shift_range, shift_range, shift_range, shift_range), fill=0)
            image = image.crop((shift_range-dx, shift_range-dy, shift_range-dx+self.image_size[0], shift_range-dy+self.image_size[1]))
            mask = mask.crop((shift_range-dx, shift_range-dy, shift_range-dx+self.image_size[0], shift_range-dy+self.image_size[1]))

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask

    def rgb_to_class(self, mask):
        mask = (mask >= 0.5).long()
        mask = 4 * mask[0] + 2 * mask[1] + mask[2]

        class_mask = torch.zeros_like(mask, dtype=torch.long)
        class_mask[mask == 3] = 0  # Urban land (Cyan: 011)
        class_mask[mask == 6] = 1  # Agriculture land (Yellow: 110)
        class_mask[mask == 5] = 2  # Rangeland (Purple: 101)
        class_mask[mask == 2] = 3  # Forest land (Green: 010)
        class_mask[mask == 1] = 4  # Water (Blue: 001)
        class_mask[mask == 7] = 5  # Barren land (White: 111)
        class_mask[mask == 0] = 6  # Unknown (Black: 000)

        return class_mask

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # Get the probability of the true class for each example
        pt = torch.exp(-ce_loss)

        # Focal loss calculation
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# Create datasets and data loaders
train_dataset = SegmentationDataset(root_dir='/home/hw1_data/p2_data/train', image_size=(512, 512), augment=True, oversample_factor=6)
val_dataset = SegmentationDataset(root_dir='/home/hw1_data/p2_data/validation', image_size=(512, 512), augment=False)
batch_size = 30
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Initialize the model, loss function, and optimizer
model = create_model(num_classes=7, pretrained=True).to(device)
alpha = torch.tensor([2.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]).to(device)
criterion = FocalLoss(alpha=alpha, gamma=2.0).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

# Learning rate scheduler
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

def mean_iou_score(pred, labels):
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp + 1e-8)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)
    return mean_iou

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss

def validate(model, dataloader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)['out']
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = running_loss / num_batches
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    mean_iou = mean_iou_score(all_preds, all_labels)

    return avg_loss, mean_iou

# Training loop
num_epochs = 100
best_val_loss = float('inf')
best_val_iou = 0.0
save_dir = '/home/checkpoints_DeepLabV3Plus'
os.makedirs(save_dir, exist_ok=True)

train_losses, val_losses, val_ious = [], [], []

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")
    
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_iou = validate(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    
    # Learning rate scheduling
    # scheduler.step(val_loss)
    
    # Save checkpoints
    if epoch == 1:
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_first_epoch.pth'))
    elif epoch == num_epochs // 2:
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_middle_epoch.pth'))
    elif epoch == num_epochs:
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_last_epoch.pth'))
    
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_best_iou.pth'))

print("Training complete!")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Training and Validation Loss\nBest Val Loss: {best_val_loss:.4f}')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_ious, label='Val IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title(f'Validation IoU\nBest Val IoU: {best_val_iou:.4f}')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_curves.png'))
plt.close()