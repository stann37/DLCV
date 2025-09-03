import torch
import lpips
# pip install lpips
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def compute_lpips(gt_folder, test_folder):
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net='vgg').cuda()
    
    # Setup image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])
    ])
    
    # Get all image files
    gt_files = [f for f in sorted(os.listdir(gt_folder)) 
                if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    total_lpips = 0.0
    count = 0
    
    # Create progress bar
    pbar = tqdm(gt_files, desc="Computing LPIPS")
    for img_name in pbar:
        # Load images
        gt_path = os.path.join(gt_folder, img_name)
        test_path = os.path.join(test_folder, img_name)
        
        if not os.path.exists(test_path):
            pbar.write(f"Warning: Missing test image for {img_name}")
            continue
            
        gt_img = Image.open(gt_path).convert('RGB')
        test_img = Image.open(test_path).convert('RGB')
        
        # Transform images
        gt_tensor = transform(gt_img).unsqueeze(0).cuda()
        test_tensor = transform(test_img).unsqueeze(0).cuda()
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_value = loss_fn(gt_tensor, test_tensor).item()
            
        total_lpips += lpips_value
        count += 1
        
        # Update progress bar with current average
        current_avg = total_lpips / count
        pbar.set_postfix({'avg_lpips': f'{current_avg:.4f}'})
    
    # Compute final average LPIPS
    avg_lpips = total_lpips / count if count > 0 else float('nan')
    print(f"\nFinal Average LPIPS: {avg_lpips:.4f}")
    print(f"Total images processed: {count}")
    return avg_lpips

if __name__ == "__main__":
    
    test_folder = "test21"
    gt_folder = "dataset/public_test/images"
    
    lpips_score = compute_lpips(gt_folder, test_folder)