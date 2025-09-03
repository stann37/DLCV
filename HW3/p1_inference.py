import json
import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load LLaVA model and processor"""
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    # Initialize model with float16 precision
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def get_image_prediction(image_path, model, processor):
    """Generate caption for a single image"""
    
    # Load and preprocess image
    raw_image = Image.open(image_path).convert('RGB')
    
    instruction = "Write a description for the photo in one sentence."
    prompt = f"USER: <image>\n{instruction}. ASSISTANT:"
    
    # Apply chat template
    # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Prepare inputs
    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors='pt'
    ).to(device, torch.float16)
    
    # Generate caption
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
    
    # Decode output
    caption = processor.decode(output[0][2:], skip_special_tokens=True)
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:")[-1].strip()
    
    return caption

def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python3 p1_inference.py <image_folder_path> <output_json_path>")
        sys.exit(1)
    
    # Get input and output paths from command line arguments
    image_folder = sys.argv[1]
    output_path = sys.argv[2]
    
    # Verify input folder exists
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' does not exist")
        sys.exit(1)
    
    # Load model
    model, processor = load_model()
    
    # Dictionary to store results
    results = {}
    
    # Process all images in input folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct full image path
            image_path = os.path.join(image_folder, filename)
            
            # Get prediction
            caption = get_image_prediction(image_path, model, processor)
            
            # Store result using filename without extension as key
            key = os.path.splitext(filename)[0]
            results[key] = caption
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save results to json file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()