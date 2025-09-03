import os
import sys
sys.path.append("stable-diffusion/src")
import json
import csv
from PIL import Image
import torch
import clip
from tqdm import tqdm

# Assuming CLIP is installed and the necessary paths are set up
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the id2label mapping
with open('hw2_data/clip_zeroshot/id2label.json', 'r') as f:
    id2label = json.load(f)

# Prepare the text inputs
text_inputs = torch.cat([clip.tokenize(f"A photo of {label}") for label in id2label.values()]).to(device)

# Encode the text inputs
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Function to predict the label for an image
def predict_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity[0].topk(1)

# Process all images in the validation set
val_dir = 'hw2_data/clip_zeroshot/val'
results = []
correct = 0
total = 0

for filename in tqdm(os.listdir(val_dir)):
    if filename.endswith('.png'):
        image_path = os.path.join(val_dir, filename)
        true_class_id = filename.split('_')[0]
        true_label = id2label[true_class_id]
        
        value, index = predict_image(image_path)
        predicted_label = list(id2label.values())[index]
        
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'filename': filename,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': value.item(),
            'is_correct': is_correct
        })

# Calculate accuracy
accuracy = correct / total

# Save results to CSV
with open('clip_zeroshot_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'true_label', 'predicted_label', 'confidence', 'is_correct'])
    writer.writeheader()
    writer.writerows(results)

print(f"Accuracy: {accuracy:.2%}")

# Report 5 successful and 5 failed cases
successful = [r for r in results if r['is_correct']][:5]
failed = [r for r in results if not r['is_correct']][:5]

print("\n5 Successful cases:")
for case in successful:
    print(f"File: {case['filename']}, True: {case['true_label']}, Predicted: {case['predicted_label']}")

print("\n5 Failed cases:")
for case in failed:
    print(f"File: {case['filename']}, True: {case['true_label']}, Predicted: {case['predicted_label']}")
    
# 58.56%