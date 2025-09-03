import torch
import os
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image

from p1_model import ContextUnet, DDPM

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
save_denoise = False

# Custom save_image function as specified
def save_image(tensor, filepath):
    tensor = tensor.clamp(0, 1) * 255
    tensor = tensor.to(torch.uint8)
    img_np = tensor.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
    img_pil = Image.fromarray(img_np, 'RGB')
    img_pil.save(filepath)

def generate_and_save_images(model, output_folder, n_samples=50, image_size=(3, 28, 28), device='cuda', guide_w=35):
    mnistm_dir = os.path.join(output_folder, 'mnistm')
    svhn_dir = os.path.join(output_folder, 'svhn')
    os.makedirs(mnistm_dir, exist_ok=True)
    os.makedirs(svhn_dir, exist_ok=True)
    
    # Only create denoise directory if save_denoise is True
    if save_denoise:
        denoise_dir = os.path.join(os.getcwd(), 'denoise')
        os.makedirs(denoise_dir, exist_ok=True)

    first_zero_mnistm = first_zero_svhn = True

    with torch.no_grad():
        for digit in tqdm(range(10), desc="Generating digits", unit="digit"):
            for dataset, dataset_name in [(0, 'mnistm'), (1, 'svhn')]:
                combined_class = digit + 10 * dataset
                save_intermediate = (digit == 0 and 
                                     ((dataset_name == 'mnistm' and first_zero_mnistm) or 
                                      (dataset_name == 'svhn' and first_zero_svhn)))

                samples, intermediate_steps = model.sample(
                    n_sample=n_samples,
                    size=image_size,
                    device=device,
                    combined_class=combined_class,
                    guide_w=guide_w,
                    save_intermediate=save_intermediate
                )

                for i in range(n_samples):
                    img = samples[i]
                    img = (img - img.min()) / (img.max() - img.min())
                    filename = f"{digit}_{i+1:03}.png"
                    output_path = os.path.join(mnistm_dir if dataset == 0 else svhn_dir, filename)
                    save_image(img, output_path)

                # Only save denoise steps if save_denoise is True
                if save_denoise and save_intermediate and intermediate_steps is not None:
                    for timestep, step in intermediate_steps:
                        denoise_img = torch.tensor(step[0]).to(device)
                        denoise_img = (denoise_img - denoise_img.min()) / (denoise_img.max() - denoise_img.min())
                        if denoise_img.ndim == 3 and denoise_img.shape[0] == 1:
                            denoise_img = denoise_img.repeat(3, 1, 1)
                        elif denoise_img.ndim == 3 and denoise_img.shape[0] != 3:
                            denoise_img = denoise_img.permute(2, 0, 1)
                        
                        denoise_filename = f"{dataset_name}_0_{timestep}.png"
                        denoise_output_path = os.path.join(denoise_dir, denoise_filename)
                        save_image(denoise_img, denoise_output_path)

                    if dataset_name == 'mnistm':
                        first_zero_mnistm = False
                    elif dataset_name == 'svhn':
                        first_zero_svhn = False
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conditional digit images.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder")
    args = parser.parse_args()
    
    # Initialize and load the model
    n_classes = 20
    n_T = 750  # High number of timesteps for quality
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=128, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0)
    ddpm.load_state_dict(torch.load("p1_model.pth", map_location=device))
    
    ddpm.to(device)

    # Generate and save images
    generate_and_save_images(ddpm, args.output_folder, device=device)