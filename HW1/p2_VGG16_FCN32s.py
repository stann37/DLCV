import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 7

class VGG16_FCN32s(nn.Module):
    def __init__(self):
        super(VGG16_FCN32s, self).__init__()
        self.VGG16 = models.vgg16(pretrained=True)
        self.features = self.VGG16.features

        # Freeze VGG16 layers
        #for param in self.features.parameters():
            #param.requires_grad = False

        # FCN layers
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),  # fc6
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),  # fc7
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)  # score_fr
        )
        
        # 32x upsampling
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.fcn(x)
        x = self.upscore(x)
        return x

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(512, 512), augment=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith('_sat.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        # Load image and mask
        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Apply transformations
        if self.augment:
            image, mask = self.transform(image, mask)
        else:
            image = TF.resize(image, self.image_size)
            mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask = TF.to_tensor(mask)

        # Convert mask to categorical
        mask = self.rgb_to_class(mask)

        return image, mask

    def transform(self, image, mask):
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
        
        # Random rotation
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        
        # Color jitter (only for image)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize (only for image)
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

# Create datasets and data loaders
train_dataset = SegmentationDataset(root_dir='./hw1_data/p2_data/train', image_size=(512, 512), augment=True)
val_dataset = SegmentationDataset(root_dir='./hw1_data/p2_data/validation', image_size=(512, 512), augment=False)
batch_size = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the model, loss function, and optimizer
model = VGG16_FCN32s().to(device)
class_weights = torch.tensor([1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 2.0]).to(device)  # Increase weight for class 2 (Rangeland)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Gradient clipping
max_grad_norm = 1.0

# Metrics
def compute_iou(pred, labels, num_classes):
    pred = pred.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    iou_list = []
    
    for cls in range(num_classes):
        intersection = np.sum((pred == cls) & (labels == cls))
        union = np.sum((pred == cls) | (labels == cls))
        iou = intersection / union if union > 0 else float('nan')
        iou_list.append(iou)

    iou_list = [iou for iou in iou_list if not np.isnan(iou) and iou is not None]
    mean_iou = np.nanmean(iou_list)
    return mean_iou, iou_list

# Training and validation functions
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = len(dataloader)

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        running_loss += loss.item()
        mean_iou, _ = compute_iou(outputs, labels, num_classes)
        running_iou += mean_iou

    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    return avg_loss, avg_iou

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    iou_per_class = [0] * num_classes
    num_batches = len(dataloader)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            mean_iou, batch_iou_per_class = compute_iou(outputs, labels, num_classes)
            running_iou += mean_iou
            iou_per_class = [sum(x) for x in zip(iou_per_class, batch_iou_per_class)]

    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    avg_iou_per_class = [iou / num_batches for iou in iou_per_class]
    return avg_loss, avg_iou, avg_iou_per_class

# Training loop
num_epochs = 150
best_val_loss = float('inf')
best_val_iou = 0.0
save_dir = './checkpoints_VGG16_FCN32s'
os.makedirs(save_dir, exist_ok=True)

# Lists to store metrics for plotting
train_losses, train_ious = [], []
val_losses, val_ious = [], []

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")
    
    # Train and validate
    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_iou, val_iou_per_class = validate(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
    print(f"Val IoU per class: {[f'{iou:.4f}' for iou in val_iou_per_class]}")
    
    # Store metrics for plotting
    train_losses.append(train_loss)
    train_ious.append(train_iou)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    
    # Save checkpoints
    if epoch == 1:
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_first_epoch.pth'))
    elif epoch == num_epochs // 2:
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_middle_epoch.pth'))
    elif epoch == num_epochs:
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_last_epoch.pth'))
    
    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_best_loss.pth'))
    
    # Save best model based on validation IoU
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_best_iou.pth'))

print("Training complete!")

# Plotting
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Training and Validation Loss\nBest Val Loss: {best_val_loss:.4f}')

# IoU plot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_ious, label='Train IoU')
plt.plot(range(1, num_epochs + 1), val_ious, label='Val IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title(f'Training and Validation IoU\nBest Val IoU: {best_val_iou:.4f}')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_curves.png'))
plt.close()