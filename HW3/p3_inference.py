import os
import json
import math
import random
import copy
import collections
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform
import loralib as lora
import matplotlib.pyplot as plt
from tqdm import tqdm
from tokenizer import BPETokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import CLIPModel
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        
        self.lora_r = 32
        self.lora_alpha = 64

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Instead of separate Q,K,V layers
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=cfg.lora_r, lora_alpha=cfg.lora_alpha)  # Single layer for Q,K,V
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=cfg.lora_r, lora_alpha=cfg.lora_alpha)
        
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, output_attentions=False):
        B, T, C = x.size() # batch, context, embedding
        
        # Project to q,k,v all at once then split
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each [B, T, C]
        
        # Reshape to support multiple heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
        
        # Compute attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, nh, T, T]
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        
        if output_attentions:
            return out, att  # Return attention weights too
        return out

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=cfg.lora_r, lora_alpha=cfg.lora_alpha)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=cfg.lora_r, lora_alpha=cfg.lora_alpha))
        ]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
            

    def forward(self, image_embeddings, tokens):
        # Convert tokens to embeddings
        token_embeddings = self.transformer.wte(tokens)  # [B*num_caps, T, n_embd]
        
        # Concatenate image embeddings with token embeddings
        x = torch.cat([image_embeddings, token_embeddings], dim=1)  # [B*num_caps, T+257, n_embd]
        
        # Get positional embeddings
        pos = torch.arange(x.size(1), dtype=torch.long, device=x.device)  # [T+257]
        pos = pos.unsqueeze(0).expand(x.size(0), -1)  # [B*num_caps, T+257]
        pos_emb = self.transformer.wpe(pos)  # [B*num_caps, T+257, n_embd]
        
        # Add positional embeddings
        x = x + pos_emb
        
        # Pass through transformer
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
        return x
    
class ImageCaptioningModel(nn.Module):
    def __init__(self, cfg = "hw3_data/p2_data/decoder_model.bin"):
        super().__init__()
        
        self.tokenizer = BPETokenizer("encoder.json","vocab.bpe")
        self.start_token = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        # Vision encoder - get all patch features including CLS token
        loaded_vit=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_encoder =loaded_vit.vision_model 
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # Project from 1024 to decoder dimension
        self.projection = nn.Linear(1024, 768)
        
        self.cfg = Config(cfg)
        self.decoder = Decoder(self.cfg).to(device)
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        # Enable projection parameters
        for param in self.projection.parameters():
            param.requires_grad = True

        # Enable LoRA parameters
        for name, param in self.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
    def forward(self):
        return 0
    

def test_attention_weights(model, img_path):
    # Load and process image
    config = resolve_data_config({}, model='vit_large_patch14_clip_224.openai_ft_in12k_in1k')
    transform = create_transform(**config)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Get visual features
    with torch.no_grad():
        visual_features = model.vision_encoder(img_tensor).last_hidden_state
        visual_features = model.projection(visual_features)
        print(f"Visual features shape: {visual_features.shape}")

        # Initialize with start token
        current_tokens = torch.tensor([[model.start_token]], device=device)
        attention_maps = []
        generated_tokens = []
        
        for step in range(30):
            # Get model outputs
            logits = model.decoder(visual_features, current_tokens)
            next_token_logits = logits[:, -1, :]  
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated_tokens.append(next_token.item())
            
            # Get attention weights
            token_embeddings = model.decoder.transformer.wte(current_tokens)
            x = torch.cat([visual_features, token_embeddings], dim=1)
            
            # Add positional embeddings  
            pos = torch.arange(x.size(1), dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = model.decoder.transformer.wpe(pos)
            x = x + pos_emb
            
            x = torch.narrow(x, 1, 0, min(x.size(1), model.decoder.block_size))
            
            # Get attention from last two layers
            last_block = model.decoder.transformer.h[-1]
            second_last_block = model.decoder.transformer.h[-2]
            
            # Process last layer
            x_last = last_block.ln_1(x)
            _, attention_weights_last = last_block.attn(x_last, output_attentions=True)
            
            # Process second last layer 
            x_second = second_last_block.ln_1(x)
            _, attention_weights_second = second_last_block.attn(x_second, output_attentions=True)
            
            # Multiple options for attention aggregation:
            
            # Option 1: Max across heads from last layer
            attention_last_max = attention_weights_last[0, :, -1, 1:257].max(dim=0)[0].reshape(16, 16)
            
            # Option 2: Mean across heads from last layer 
            attention_last_mean = attention_weights_last[0, :, -1, 1:257].mean(dim=0).reshape(16, 16)
            
            # Option 3: Average of last and second last layers (using max across heads)
            attention_second_max = attention_weights_second[0, :, -1, 1:257].max(dim=0)[0].reshape(16, 16)
            attention_combined = (attention_last_max + attention_second_max) / 2

            # Choose which attention map to use
            # attention_maps.append(attention_last_max.detach())  # Option 1
            attention_maps.append(attention_last_mean.detach())  # Option 2 
            # attention_maps.append(attention_combined.detach())  # Option 3
            
            # Print stats for analysis
            token_text = model.tokenizer.decode([next_token.item()])
            # print(f"\nToken: {token_text}")
            # print(f"Last layer max attention - min: {attention_last_max.min():.4f}, max: {attention_last_max.max():.4f}, mean: {attention_last_max.mean():.4f}")
            # print(f"Last layer mean attention - min: {attention_last_mean.min():.4f}, max: {attention_last_mean.max():.4f}, mean: {attention_last_mean.mean():.4f}")
            # print(f"Combined layers attention - min: {attention_combined.min():.4f}, max: {attention_combined.max():.4f}, mean: {attention_combined.mean():.4f}")

            # Add token and check for EOS
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(-1)], dim=1)
            if next_token.item() == model.start_token:
                break

        return attention_maps, generated_tokens

def save_attention_heatmaps(img_path, attention_maps, tokens, save_path, tokenizer):
    # Load and transform the image
    config = resolve_data_config({}, model="vit_large_patch14_clip_224.openai_ft_in12k_in1k")
    transform = create_transform(**config)
    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # Save original image size (width, height)
    img_tensor = transform(img).unsqueeze(0) 

    # Convert tokens to words
    decoded_tokens = []
    current_token = ""
    for t in tokens:
        word = tokenizer.decode([t])
        # if word == "<|endoftext|>":
        #     word = "EOS"
        decoded_tokens.append(word)
        
    # Prepare grid for visualization including original image
    num_subplots = len(tokens) + 1  # +1 for original image
    num_cols = 5
    num_rows = (num_subplots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows))
    axes = axes.flatten()
    
    # Plot original image first
    axes[0].imshow(img)
    axes[0].set_title("<start>", fontsize = 32)
    axes[0].axis("off")

    # Plot attention heatmaps
    for idx, (attention_map, token) in enumerate(zip(attention_maps, decoded_tokens)):
        ax = axes[idx + 1]  # +1 because original image takes first spot
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.cpu().numpy()

        # Resize attention map to original image size
        attention_map_resized = Image.fromarray(attention_map).resize(original_size, Image.BILINEAR)
        attention_map_resized_np = np.array(attention_map_resized)

        # Normalize the attention weights
        if attention_map_resized_np.max() != attention_map_resized_np.min():
            attention_map_resized_np = (attention_map_resized_np - attention_map_resized_np.min()) / (
                attention_map_resized_np.max() - attention_map_resized_np.min()
            )

        # Overlay heatmap onto image
        ax.imshow(img)  # Original image
        heatmap = ax.imshow(attention_map_resized_np, cmap="jet", alpha=0.5)  # Heatmap overlay
        ax.set_title(f'{token}', fontsize=32)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_subplots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
def main(): 
    print("Loading model...")
    model = ImageCaptioningModel("hw3_data/p2_data/decoder_model.bin").to(device)
    checkpoint = torch.load('p2_best.pth')
    model_state_dict = model.state_dict()
    for k, v in checkpoint['model_state_dict'].items():
        if "lora_" in k or "projection" in k:
            model_state_dict[k] = v
    model.load_state_dict(model_state_dict)
    model.eval()
    os.makedirs("p3_images", exist_ok=True)
    
    mode = "mean"
    
    # part 1
    images = ["bike", "girl", "sheep", "ski", "umbrella"]
    for image_str in images:
        img_path = f"hw3_data/p3_data/images/{image_str}.jpg"
        attention_maps, tokens = test_attention_weights(model, img_path)
        save_attention_heatmaps(img_path, attention_maps, tokens, 
                            os.path.join("p3_images", f"{image_str}_{mode}.jpg"),
                            model.tokenizer)
    
    # part 2
    max_name = "000000001086"
    img_path = f"hw3_data/p2_data/images/val/{max_name}.jpg"
    attention_maps, tokens = test_attention_weights(model, img_path)
    save_attention_heatmaps(img_path, attention_maps, tokens,
                          os.path.join("p3_images", f"max_{mode}.jpg"),
                          model.tokenizer)
    
    min_name = "000000000679" 
    img_path = f"hw3_data/p2_data/images/val/{min_name}.jpg"
    attention_maps, tokens = test_attention_weights(model, img_path)
    save_attention_heatmaps(img_path, attention_maps, tokens,
                          os.path.join("p3_images", f"min_{mode}.jpg"), 
                          model.tokenizer)
    
if __name__ == "__main__":
    main()