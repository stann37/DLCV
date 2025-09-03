import os
import sys
import torch
from UNet import UNet
from p2_DDIM import DDIM
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

def slerp(val, low, high):
    low_norm = torch.norm(low)
    high_norm = torch.norm(high)
    
    # Normalize the input tensors
    low_normalized = low / low_norm
    high_normalized = high / high_norm
    
    # Compute cosine of the angle between the two vectors
    dot_product = torch.sum(low_normalized * high_normalized)
    
    # Clamp the dot product to avoid numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute the angle (theta) between vectors
    theta = torch.acos(dot_product)
    
    # Compute sin(theta) for later use
    sin_theta = torch.sin(theta)
    
    # Compute the SLERP
    if sin_theta > 1e-6:  # Avoid division by zero
        return (
            torch.sin((1 - val) * theta) / sin_theta * low +
            torch.sin(val * theta) / sin_theta * high
        )
    else:
        # If the angle is very small, fall back to linear interpolation
        return (1 - val) * low + val * high

def linear_interpolation(val, low, high):
    return (1.0 - val) * low + val * high

def main():
    noise_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ddim = DDIM(nn_model=model, device=device)
    #print(ddim.timesteps)
    #print(ddim.betas)
    #print(ddim.alphas)
    noise_files = sorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.pt')])

    # Commented part for individual noise file processing
    
    for i, noise_file in enumerate(noise_files):
        noise = torch.load(noise_file).to(device)
        
        # Ensure noise has a batch dimension: [B, C, H, W]
        if noise.dim() == 3:  
            noise = noise.unsqueeze(0)  # Add batch dimension if missing

        # Perform DDIM sampling
        sampled_image = ddim.sample(noise=noise, ddim_timesteps=50, ddim_eta=0.0)
        # print(sampled_image.max(), " ", sampled_image.min())

        # Min-max normalization to [0, 1]
        min_val = sampled_image.min()
        max_val = sampled_image.max()
        sampled_image = (sampled_image - min_val) / (max_val - min_val)

        # Save output image
        save_image(sampled_image, os.path.join(output_dir, f"{i:02d}.png"))
    

    if False:
        output_dir = 'p2_images'
        
        # Commented part for eta experiments
        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for eta in etas:
            eta_dir = os.path.join(output_dir, f"eta_{eta}")
            os.makedirs(eta_dir, exist_ok=True)
            for i in range(4):  # Assume testing for the first 4 noise files
                noise_file = noise_files[i]
                noise = torch.load(noise_file).to(device)
                
                # Ensure noise has a batch dimension: [B, C, H, W]
                if noise.dim() == 3:
                    noise = noise.unsqueeze(0)

                sampled_image = ddim.sample(noise=noise, ddim_timesteps=50, ddim_eta=eta)
                save_image((sampled_image + 1) / 2, os.path.join(eta_dir, f"{i:02d}.png"))

        # Interpolation between two noise files using slerp and linear interpolation
        interp_dir = os.path.join(output_dir, "interpolation")
        os.makedirs(interp_dir, exist_ok=True)
        
        # Load and ensure correct dimensions for noise1 and noise2
        noise1 = torch.load(noise_files[0]).to(device)
        noise2 = torch.load(noise_files[1]).to(device)
        if noise1.dim() == 3:
            noise1 = noise1.unsqueeze(0)
        if noise2.dim() == 3:
            noise2 = noise2.unsqueeze(0)

        # Remove batch dimensions for interpolation calculations
        noise1 = noise1.squeeze(0)
        noise2 = noise2.squeeze(0)
        
        for alpha in np.linspace(0, 1, 11):
            slerp_interp = slerp(alpha, noise1, noise2).unsqueeze(0)  # Re-add batch dimension
            linear_interp = linear_interpolation(alpha, noise1, noise2).unsqueeze(0)  # Re-add batch dimension
            
            slerp_image = ddim.sample(noise=slerp_interp, ddim_timesteps=50, ddim_eta=0.0)
            linear_image = ddim.sample(noise=linear_interp, ddim_timesteps=50, ddim_eta=0.0)
            
            save_image((slerp_image + 1) / 2, os.path.join(interp_dir, f"slerp_{alpha:.1f}.png"))
            save_image((linear_image + 1) / 2, os.path.join(interp_dir, f"linear_{alpha:.1f}.png"))

if __name__ == "__main__":
    main()