import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from byol_pytorch import BYOL
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# HYPER-PARAMETERS
num_epochs = 300
batch_size = 256
lr = 0.001
weight_decay = 0.01

# Initialize the model and BYOL learner
model = torchvision.models.resnet50(weights=None).to(device)
learner = BYOL(
    model,
    image_size=128,
    hidden_layer='avgpool'
).to(device)

opt = torch.optim.AdamW(learner.parameters(), lr=lr, weight_decay=weight_decay)
#opt = torch.optim.SGD(learner.parameters(), lr=0.3, momentum=0.9, weight_decay=5e-7)

# Dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset and dataloader
mini_dataset = CustomImageDataset('./hw1_data/p1_data/mini/train', transform=transform)
print(f"Number of images found: {len(mini_dataset)}")
mini_dataloader = DataLoader(mini_dataset, batch_size=batch_size, shuffle=True)

checkpoint_dir = './checkpoints_resnet50'
final_model_path = os.path.join(checkpoint_dir, 'improved-net.pt')

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Track loss for plotting and lowest loss
epoch_losses = []
best_loss = float('inf')  # Initialize best loss to infinity

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    for images in tqdm(mini_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        images = images.to(device)

        loss = learner(images)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()  # Update moving average of target encoder

        epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(mini_dataloader)
    epoch_losses.append(avg_epoch_loss)
    
    # Print average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

    # Save checkpoint if the current loss is the lowest so far
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model_path = os.path.join(checkpoint_dir, 'resnet_best_model.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': avg_epoch_loss,
        }, best_model_path)
        print(f'Best model saved at: {best_model_path}')

    # Save checkpoint at the last epoch
    if epoch == num_epochs - 1:
        last_model_path = os.path.join(checkpoint_dir, 'resnet_last_model.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': avg_epoch_loss,
        }, last_model_path)
        print(f'Last model saved at: {last_model_path}')

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)

# Save the loss plot
plt.savefig(os.path.join(checkpoint_dir, 'training_loss_plot.png'))
plt.close()  # Optionally display the plot
