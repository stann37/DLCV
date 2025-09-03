import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models.segmentation as models
from PIL import Image
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def create_model(num_classes, pretrained=True):
    model = models.deeplabv3_resnet50(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def save_mask(mask, filename):
    color_map = {
        0: [0, 255, 255],  # Urban land (Cyan)
        1: [255, 255, 0],  # Agriculture land (Yellow)
        2: [255, 0, 255],  # Rangeland (Purple)
        3: [0, 255, 0],    # Forest land (Green)
        4: [0, 0, 255],    # Water (Blue)
        5: [255, 255, 255],# Barren land (White)
        6: [0, 0, 0]       # Unknown (Black)
    }
    
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb_mask[mask == class_id] = color
    
    img = Image.fromarray(rgb_mask)
    img.save(filename)

def main(input_dir, output_dir):
    # Set up paths
    model_path = 'checkpoint_p2.pth'  # Update this path if necessary

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_model(num_classes=7, pretrained=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Process input images
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('_sat.jpg'):
            # Load and preprocess the image
            image_path = os.path.join(input_dir, filename)
            input_tensor = load_image(image_path).to(device)

            # Perform inference
            with torch.no_grad():
                output = model(input_tensor)['out']
            
            # Post-process the output
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

            # Save the mask
            mask_filename = filename.replace('_sat.jpg', '_mask.png')
            save_mask(pred_mask, os.path.join(output_dir, mask_filename))

    print(f"Inference complete. Masks saved in {output_dir}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 inference.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)