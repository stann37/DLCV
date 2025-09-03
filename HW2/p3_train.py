import sys
import os
sys.path.append("stable-diffusion")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm import LatentDiffusion

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def train_concept(model, tokenizer, concept_image_folder, placeholder_token, init_word, num_vectors=4, num_epochs=1000, batch_size=4, is_style=False):
    # Initialize placeholder tokens
    placeholder_tokens = [f"{placeholder_token}-{i}" for i in range(num_vectors)]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    
    model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))
    
    # Initialize the new token embeddings
    token_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight.data
    init_word_ids = tokenizer.encode(init_word)
    for token in placeholder_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_embeds[token_id] = token_embeds[init_word_ids[1]].clone().detach().requires_grad_(True)

    # Prepare dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.images = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.images[idx])
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

    dataset = CustomDataset(concept_image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Training loop
    optimizer = torch.optim.AdamW(model.cond_stage_model.transformer.get_input_embeddings().parameters(), lr=5e-4, betas=(0.9, 0.999))
    
    losses = []
    
    # Text templates
    if is_style:
        templates = [
            'an illustration in the style of {}',
            'a rendering in the style of {}',
            'the illustration in the style of {}',
            'a clean illustration in the style of {}',
            'a picture in the style of {}',
            'a cool illustration in the style of {}',
            'a bright illustration in the style of {}',
            'a good illustration in the style of {}',
            'a rendition in the style of {}',
            'a nice illustration in the style of {}',
        ]
    else:
        templates = [
            "a photo of {}",
            "a rendering of {}",
            "the photo of {}",
            "a picture of {}",
            "a cool photo of {}",
            "a close-up photo of {}",
            "a bright photo of {}",
            "a good photo of {}",
            "a close-up photo of {}",
            "a rendition of {}",
            "a nice photo of {}",
        ]

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Generate prompts for each image in the batch
            prompts = [random.choice(templates).format(" ".join(placeholder_tokens)) for _ in range(batch.shape[0])]
            
            # Encode the image to latent space
            with torch.no_grad():
                latent = model.get_first_stage_encoding(model.encode_first_stage(batch.to(model.device)))
            
            # Get text encoder hidden states
            encoder_hidden_states = model.get_learned_conditioning(prompts)
            
            # Forward pass
            t = torch.randint(0, model.num_timesteps, (latent.shape[0],), device=model.device).long()
            noise = torch.randn_like(latent)
            noisy_latent = model.q_sample(x_start=latent, t=t, noise=noise)
            
            # Predict the noise
            noise_pred = model.apply_model(noisy_latent, t, encoder_hidden_states)
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

        # Save intermediate embeddings every 100 epochs
        if (epoch + 1) % 100 == 0:
            os.makedirs("textual_inversion_embeddings", exist_ok=True)
            learned_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight[tokenizer.convert_tokens_to_ids(placeholder_tokens)].detach().cpu()
            torch.save(learned_embeds, f"textual_inversion_embeddings/{placeholder_token[1:-1]}_epoch{epoch+1}.pt")

    # Save the final learned embeddings
    learned_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight[tokenizer.convert_tokens_to_ids(placeholder_tokens)].detach().cpu()
    torch.save(learned_embeds, f"textual_inversion_embeddings/{placeholder_token[1:-1]}_final.pt")

    # Plot and save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f'Training Loss for {placeholder_token}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.savefig(f'loss_curve_{placeholder_token[1:-1]}.png')
    plt.close()

def main():
    config_path = "stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    ckpt_path = "stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt"
    
    concept1_folder = "hw2_data/textual_inversion/0"
    concept1_token = "<new1>"
    concept1_init = "corgi dog"
    
    concept2_folder = "hw2_data/textual_inversion/1"
    concept2_token = "<new2>"
    concept2_init = "fantasy illustration."

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    tokenizer = model.cond_stage_model.tokenizer

    # Train first concept (dog)
    train_concept(model, tokenizer, concept1_folder, concept1_token, concept1_init, is_style=False)

    # Train second concept (art style)
    train_concept(model, tokenizer, concept2_folder, concept2_token, concept2_init, is_style=True)

if __name__ == "__main__":
    main()