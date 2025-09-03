import os
import csv
import sys
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define the number of classes (adjust as needed)
num_classes = 65  # Assuming 65 classes based on the given model

class Classifier(torch.nn.Module):
    def __init__(self):
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
        
    def forward(self, x):
        return self.backbone(x)

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main(csv_path, img_folder, output_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier().to(device)
    
    checkpoint = torch.load("checkpoint_p1.pth", map_location=device)
    
    # Check if the checkpoint contains a state_dict or if it's already a state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Read the input CSV file
    df = pd.read_csv(csv_path)

    # Prepare the output CSV file
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'filename', 'label'])

        # Process each image
        for index, row in df.iterrows():
            img_path = os.path.join(img_folder, row['filename'])
            img = Image.open(img_path).convert('RGB')
            img_tensor = val_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)

            writer.writerow([row['id'], row['filename'], predicted.item()])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python p1_inference.py <path_to_csv> <path_to_image_folder> <path_to_output_csv>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])