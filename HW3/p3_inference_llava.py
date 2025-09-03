import json
import os
import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import CLIPProcessor, CLIPVisionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load LLaVA model and processor"""
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        output_attentions=True,
        return_dict_in_generate=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def get_image_prediction_with_attention(image_path, model, processor):
    raw_image = Image.open(image_path).convert('RGB')
    prompt = f"USER: <image>\nWrite a description for the photo in one sentence. ASSISTANT:"
    
    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors='pt'
    ).to(device, torch.float16)
    
    # print(inputs.keys)
    # raise Exception
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )
        
        input_length = inputs['input_ids'].size(1)
        tokens = outputs.sequences[0][input_length:]
        
        # Get patch region - start at position 7 (after <image> and empty token)
        image_token_pos = 5  # <image> token
        patch_start = image_token_pos + 2  # Start after empty token
        patch_end = patch_start + 576  # Include all patches
        
        attention_maps = []
        for token_idx in range(len(tokens)):
            # Get attention from last layer
            attention = outputs.attentions[token_idx][-1]  # [1, num_heads, query_len, key_len]
            
            # Extract attention to patches only
            patch_attention = attention[0, :, -1, patch_start:patch_end]  # [num_heads, 576]
            
            # Take max across heads for clearer visualization
            max_attn = patch_attention.mean(dim=0)  # [576]
            max_attn = max_attn.to(torch.float32).cpu()
            attention_maps.append(max_attn)
        
        attention_maps = torch.stack(attention_maps)  # [num_tokens, 576]
        
        return processor.decode(outputs.sequences[0], skip_special_tokens=False), tokens, attention_maps

def save_attention_heatmaps(image_path, attention_maps, tokens, patch_side, save_path, processor):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    decoded_tokens = [processor.decode([token_id], skip_special_tokens=False) for token_id in tokens]
    num_tokens = len(decoded_tokens)
    
    # Create subplot grid with original image
    num_cols = 5
    num_total = num_tokens + 1  # +1 for original image
    num_rows = (num_total + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4*num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot original image first
    ax = axes[0, 0]
    ax.imshow(img)
    ax.set_title("<start>", pad=5, fontsize=32)
    ax.axis('off')
    
    # Plot attention heatmaps
    for idx in range(num_tokens):
        # Calculate position accounting for original image
        pos = idx + 1  # +1 because we used first position for original
        row = pos // num_cols
        col = pos % num_cols
        ax = axes[row, col]
        
        # Reshape attention to patch grid
        attention = attention_maps[idx].reshape(patch_side, patch_side).numpy()
        
        # Normalize attention values
        if attention.max() != attention.min():
            attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        # Plot original image
        ax.imshow(img)
        
        # Resize attention map and overlay
        attention_resized = Image.fromarray(attention).resize(
            original_size, Image.BICUBIC)
        attention_resized = np.array(attention_resized)
        ax.imshow(attention_resized, cmap='jet', alpha=0.7)
        
        # Add token text
        token_text = decoded_tokens[idx].strip()
        if not token_text:
            token_text = "<space>"
        elif token_text == "</s>":
            token_text = "</s>"
        ax.set_title(token_text, pad=5, fontsize=32)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_total, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    os.makedirs("p1_attention_vis", exist_ok=True)
    
    # Load model and processor
    model, processor = load_model()
    
    # Print vision config for reference
    print("Vision Config:")
    vision_config = model.config.vision_config
    print(f"Image size: {vision_config.image_size}")
    print(f"Patch size: {vision_config.patch_size}")
    print(f"Number of patches per side: {vision_config.image_size // vision_config.patch_size}")
    print(f"Total patches: {(vision_config.image_size // vision_config.patch_size) ** 2}") # 576
    
    # Test images
    test_images = ["bike", "girl", "sheep", "ski", "umbrella"]
    # test_images = ["bike"]
    
    for image_name in test_images:
        print(f"\nProcessing image: {image_name}")
        image_path = f"hw3_data/p3_data/images/{image_name}.jpg"
        
        # Get prediction and attention maps
        caption, tokens, attention_maps = get_image_prediction_with_attention(image_path, model, processor)
        
        # Save visualization 
        save_path = f"p1_attention_vis/{image_name}_attention.jpg"
        save_attention_heatmaps(image_path, attention_maps, tokens, 24, save_path, processor)
        
        print(f"Generated caption: {caption}")
        print("-" * 50)

if __name__ == "__main__":
    # model, processor = load_model()
    
    # image = Image.open("hw3_data/p3_data/images/bike.jpg").convert("RGB")
    # prompt = "USER: <image>\nWrite a description for the photo in one sentence. ASSISTANT:"

    # # Get vision embeddings
    # vision_inputs = processor(images=image, text="", return_tensors='pt').to(device, torch.float16)
    # vision_outputs = model.vision_tower(vision_inputs['pixel_values'])
    # vision_embeddings = vision_outputs.last_hidden_state  # [1, 577, 1024]

    # # Process through full model
    # inputs = processor(
    #     images=image,
    #     text=prompt,
    #     return_tensors='pt'
    # ).to(device, torch.float16)

    # with torch.no_grad():
    #     # Get outputs with attention
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=50,
    #         output_attentions=True,
    #         return_dict_in_generate=True,
    #     )

    #     first_attn = outputs.attentions[0][-1]
    #     print("\nAnalyzing attention patterns:")
        
    #     # Test attention sums for different ranges
    #     ranges_to_test = [
    #         (6, 582, "After <image>"),
    #         (7, 583, "After empty token"),
    #     ]
        
    #     for start, end, desc in ranges_to_test:
    #         # Get attention to this region
    #         region_attn = first_attn[0, :, -1, start:end]  # [num_heads, 576]
            
    #         # Calculate statistics
    #         mean_attn = region_attn.mean().item()
    #         max_attn = region_attn.max().item()
    #         std_attn = region_attn.std().item()
            
    #         # Get attention pattern
    #         mean_pattern = region_attn.mean(dim=0)  # Average across heads
            
    #         print(f"\nRegion {desc} [{start}:{end}]:")
    #         print(f"Mean attention: {mean_attn:.6f}")
    #         print(f"Max attention: {max_attn:.6f}")
    #         print(f"Std attention: {std_attn:.6f}")
            
    #         # Plot pattern
    #         plt.figure(figsize=(15, 5))
    #         plt.subplot(2, 1, 1)
    #         plt.plot(mean_pattern.cpu().numpy())
    #         plt.title(f"Mean attention pattern for region {desc}")
            
    #         # Plot head-wise pattern for first 5 heads
    #         plt.subplot(2, 1, 2)
    #         for head in range(5):
    #             plt.plot(region_attn[head].cpu().numpy(), alpha=0.5, label=f'Head {head}')
    #         plt.legend()
    #         plt.title("Head-wise attention patterns")
            
    #         plt.savefig(f"attention_analysis_{start}_{end}.jpg")
    #         plt.close()
        
    main()