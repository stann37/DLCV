import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Using device: {device}")

num_classes = 65

# HYPERPARAMETERS
num_epochs = 300
batch_size = 64

class Classifier(torch.nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=None)
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )

        # Load pre-trained weights if a checkpoint path is provided
        if ckpt_path:
            self._load_pretrained_weights(ckpt_path)

    def _load_pretrained_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.backbone(x)

# Initialize a sample model for testing
model = Classifier(None).to(device)
#print(model)

TA_backbone = "./hw1_data/p1_data/pretrain_model_SL.pt"
pretrained_backbone = "./checkpoints_resnet50/resnet_best_model.pt"

settings = [
    {"name": "C", "ckpt_path": pretrained_backbone, "train_backbone": True},
    {"name": "A", "ckpt_path": None, "train_backbone": True},
    {"name": "B", "ckpt_path": TA_backbone, "train_backbone": True},
    {"name": "D", "ckpt_path": TA_backbone, "train_backbone": False},
    {"name": "E", "ckpt_path": pretrained_backbone, "train_backbone": False},
]

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(128, padding = 4, padding_mode = 'reflect'),  #test
    transforms.ColorJitter(brightness=0.2, contrast = 0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx]['label'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_dir = "./hw1_data/p1_data/office/train"
train_csv_file = "./hw1_data/p1_data/office/train.csv"
val_dir = "./hw1_data/p1_data/office/val"
val_csv_file = "./hw1_data/p1_data/office/val.csv"

train_dataset = CustomDataset(csv_file=train_csv_file, root_dir=train_dir, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(csv_file=val_csv_file, root_dir=val_dir, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()

def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss

def unfreeze_model(model, epoch):
    if epoch == 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.fc.parameters():
            param.requires_grad = True
    elif epoch == 8:
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
    elif epoch == 12:
        for param in model.backbone.parameters():
            param.requires_grad = True

os.makedirs("checkpoints_finetune", exist_ok=True)
results = {}


# Training Loop
for setting in settings:
    print(f"Training setting {setting['name']}")
    
    model = Classifier(setting['ckpt_path']).to(device)
    
    checkpoint_dir = f"checkpoints_finetune/{setting['name']}/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)


    val_accs = []
    val_losses = []
    train_accs = []
    train_losses = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        if setting['train_backbone']:
            unfreeze_model(model, epoch)
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.backbone.fc.parameters():
                param.requires_grad = True
        
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
        val_acc, val_loss = validate(model, val_dataloader, criterion)
        
        #scheduler.step(val_loss)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        #print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if epoch + 1 == 1 and setting['name'] == 'C':
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(checkpoint_dir, f"first_model.pth"))
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if setting['name'] == 'C':
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(checkpoint_dir, f"best_model.pth"))
        
        if epoch == num_epochs - 1 and setting['name'] == 'C':
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(checkpoint_dir, f"last_model_epoch_{epoch+1}.pth"))
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracies for {setting['name']}, best_val_acc: {best_val_acc}")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, f"training_validation_accuracies_{setting['name']}.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Losses for {setting['name']}")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, f"training_validation_losses_{setting['name']}.png"))
    plt.close()
    
    print(f"Best Validation Accuracy for setting {setting['name']}: {best_val_acc:.4f}")