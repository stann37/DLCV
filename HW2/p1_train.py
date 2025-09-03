from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image
import os
import pandas as pd
from torch.optim import AdamW

from p1_model import ContextUnet, DDPM

class DigitDataset(Dataset):
    def __init__(self, csv_file, img_dir, dataset_idx, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.dataset_idx = dataset_idx
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        label = self.labels_df.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, self.dataset_idx
    
def train_mnist():

    # hardcoding these here
    n_epoch = 100
    batch_size = 256
    n_T = 750
    device = "cuda"
    n_classes = 20
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    weight_decay = 1e-6
    save_dir = 'p1_model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset1 = DigitDataset('./hw2_data/digits/mnistm/train.csv', './hw2_data/digits/mnistm/data', 0, transform=tf)
    dataset2 = DigitDataset('./hw2_data/digits/svhn/train.csv', './hw2_data/digits/svhn/data', 1, transform=tf)
    combined_dataset = ConcatDataset([dataset1, dataset1, dataset2, dataset2]) # two parts MNIST-M
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    
    optim = AdamW(ddpm.parameters(), lr=lrate, weight_decay=weight_decay)

    best_loss = float('inf')
    loss_history = []
    for ep in range(n_epoch):
        running_loss = 0.0
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        # optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c, d in pbar:
            optim.zero_grad()
            combined_class = c + 10 * d
            combined_class = combined_class.to(device)
            
            # Move input to device
            x = x.to(device)
            
            # Forward pass and loss computation
            loss = ddpm(x, combined_class)
            loss.backward()
            
            # Update running_loss for accurate epoch average loss calculation
            running_loss += loss.item()
            
            # Update EMA of loss
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # Calculate and print average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{ep+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(save_dir, "model_best.pth")
            torch.save(ddpm.state_dict(), model_path)
            print('saved best model at ' + model_path)

        if ep == n_epoch - 1:
            last_model_path = os.path.join(save_dir, f"model_last.pth")
            torch.save(ddpm.state_dict(), last_model_path)
            print('saved last model at ' + last_model_path)
            
    plt.figure()
    plt.plot(range(1, n_epoch + 1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('training_loss_curve.png')

if __name__ == "__main__":
    train_mnist()
