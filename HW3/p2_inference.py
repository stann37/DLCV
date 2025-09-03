import os
import json
import torch
from PIL import Image
from p2_train_v2 import ImageCaptioningModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
from tqdm import tqdm
from timm.data import resolve_data_config, create_transform

device = "cuda" if torch.cuda.is_available() else "cpu"
    
class InferenceDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.config = resolve_data_config({}, model = 'vit_large_patch14_clip_224.openai_ft_in12k_in1k')
        self.transform = create_transform(**self.config)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        image = self.transform(image)
        return image, filename

def main():
    # Check arguments
    if len(sys.argv) != 4:
        print('Usage: python3 p2_inference.py <input_path> <output_path> <ckpt_path>')
        sys.exit(1)
    
    input_path = sys.argv[1]    # Directory containing images
    output_path = sys.argv[2]   # Path to output json file 
    ckpt_path = sys.argv[3]     # Path to model checkpoint
    
    print("Loading model...")
    # Model setup
    model = ImageCaptioningModel(ckpt_path).to(device)  # Pass checkpoint path to model
    model.vision_encoder = model.vision_encoder.float()
    checkpoint = torch.load('p2_best.pth')  # Load LoRA weights
    model_state_dict = model.state_dict()
    for k, v in checkpoint['model_state_dict'].items():
        if "lora_" in k or "projection" in k:
            model_state_dict[k] = v
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Create dataset and dataloader 
    print(f"Loading images from {input_path}")
    dataset = InferenceDataset(input_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=10, 
        shuffle=False,
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    # Generate predictions
    predictions = {}
    total_images = len(dataset)
    print(f"\nGenerating captions for {total_images} images...")
    
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            
            output_tokens = model.generate(images, max_length = 30)
            
            # Convert tokens to text 
            for tokens, filename in tqdm(zip(output_tokens, filenames), 
                                      desc="Processing images in batch",
                                      leave=False,
                                      total=len(filenames)):
                # Find second EOS token (skip first one)
                token_list = tokens.tolist()
                try:
                    # Get just the content between the first and second EOS token
                    first_eos_idx = token_list.index(model.start_token)
                    second_eos_idx = token_list.index(model.start_token, first_eos_idx + 1)
                    caption_tokens = token_list[first_eos_idx + 1:second_eos_idx]
                except ValueError:
                    # If no second EOS found, take everything after first EOS until the end
                    caption_tokens = token_list[first_eos_idx + 1:]
                
                # Decode and strip any extra whitespace
                caption = model.tokenizer.decode(caption_tokens).strip()
                predictions[os.path.splitext(filename)[0]] = caption
                # print(os.path.splitext(filename)[0], ": ", caption)
                
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
    
    # Save predictions
    print(f"\nSaving predictions to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()