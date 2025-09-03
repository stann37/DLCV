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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

def print_params(model):
    total_params = 0
    total_proj_params = 0
    total_lora_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            # print(f"{name}: {param.numel():,}")
            if 'projection' in name:
                total_proj_params += param.numel()
            if 'lora_' in name:
                total_lora_params += param.numel()
    trainable_params = total_lora_params + total_proj_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Projection parameters: {total_proj_params:,}")
    print(f"LoRA parameters: {total_lora_params:,}")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Trainable parameters percentage: {100 * trainable_params / total_params:.4f}%\n")

    assert trainable_params < 10_000_000
    
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

    def forward(self, x):
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
        
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.c_proj(out)

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

class ImageCaptionDataset(Dataset):
    def __init__(self, json_path, img_dir, tokenizer):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        
        self.config = resolve_data_config({}, model = 'vit_large_patch14_clip_224.openai_ft_in12k_in1k')
        self.transform = create_transform(**self.config)
        
        # Organize data differently - ensure each image has its own captions
        self.image_captions = []
        image_to_captions = {}
        
        # First collect all captions for each image
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_to_captions:
                image_to_captions[img_id] = []
            image_to_captions[img_id].append(ann['caption'])
        
        # Then match with image filenames
        for img in data['images']:
            if img['id'] in image_to_captions:
                self.image_captions.append({
                    'file_name': img['file_name'],
                    'captions': image_to_captions[img['id']]
                })
    
    def __len__(self):
        return len(self.image_captions)
    
    def __getitem__(self, idx):
        item = self.image_captions[idx]
        image_path = os.path.join(self.img_dir, item['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # print(item['captions']) match
        return image, item['captions']
    
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
        
    def forward(self, images, captions=None):
        batch_size = images.shape[0]
        
        # Get patch embeddings including CLS token
        with torch.no_grad():
            visual_features = self.vision_encoder(images).last_hidden_state  # [B, 257, 1024]
            # print(f"Visual features shape: {visual_features.shape}")  # Verify 257 embeddings
            
        visual_features = self.projection(visual_features)
         # [B, 257, 768]
        
        if captions is None:
            return visual_features
            
        num_captions = len(captions[0])
        
        # Process captions
        all_input_tokens = []
        all_target_tokens = []
        
        for i in range(batch_size):
            for j in range(num_captions):
                cap = captions[i][j]
                cap_tokens = self.tokenizer.encode(cap, allowed_special={"<|endoftext|>"})
                input_tokens = [self.start_token] + cap_tokens + [self.start_token]
                target_tokens = [self.start_token] + cap_tokens + [self.start_token]
                all_input_tokens.append(input_tokens)
                all_target_tokens.append(target_tokens)

        max_input_len = max(len(tokens) for tokens in all_input_tokens)
        # print("max_input_len: ", max_input_len) # T
        padded_input_tokens = torch.tensor([
            tokens + [self.start_token] * (max_input_len - len(tokens))
            for tokens in all_input_tokens
        ], dtype=torch.long, device=device)
        # print("shape of padded_input_tokens: ", padded_input_tokens.shape) # [B*5, T]
        
        # Expand visual features for each caption
        expanded_visual_features = []
        for i in range(batch_size):
            expanded_visual_features.append(visual_features[i:i+1].expand(num_captions, -1, -1))
        expanded_visual_features = torch.cat(expanded_visual_features, dim=0)  # [B*5, 257, 768]
        
        # Get predictions
        logits = self.decoder(expanded_visual_features, padded_input_tokens)  # [B*5, 257+T, vocab]
        # print("shape of logits: ", logits.shape)
        
        # Shift logits and targets for causal LM
        shifted_logits = logits[:, 257:-1, :]  # Start after image embeddings + first token
        shifted_targets = padded_input_tokens[:, 1:]
        
        # Create attention mask
        valid_tokens_len = [len(tokens)-1 for tokens in all_target_tokens]
        padding_mask = torch.zeros((len(all_target_tokens), shifted_targets.size(1)), device=device)
        for i, length in enumerate(valid_tokens_len):
            padding_mask[i, :length] = 1.0
            
        # Calculate loss
        shifted_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))
        shifted_targets = shifted_targets.reshape(-1)
        mask = padding_mask.reshape(-1)

        loss = F.cross_entropy(shifted_logits, shifted_targets, reduction='none')
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def greedy_decode(self, visual_features, max_length):
        batch_size = visual_features.size(0)
        
        current_tokens = torch.full((batch_size, 1), self.start_token, 
                                dtype=torch.long, device=device)
        
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length-1):
            if finished_sequences.all():
                break
                
            logits = self.decoder(visual_features, current_tokens)
            next_token_logits = logits[:, visual_features.size(1)+current_tokens.size(1)-1, :]
            
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            current_tokens = torch.cat([current_tokens, next_tokens.unsqueeze(-1)], dim=1)
            finished_sequences = finished_sequences | (next_tokens == self.start_token)
                
        return current_tokens

    def generate(self, images, max_length=30):
        
        # Get patch features
        with torch.no_grad():
            visual_features = self.vision_encoder(images).last_hidden_state
        visual_features = self.projection(visual_features)
        
        return self.greedy_decode(visual_features, max_length)

                        
def train_one_epoch(model, train_loader, optimizer, epoch, val_loader, scheduler=None, save_steps=104):
    model.train()
    
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}') 
    for step, (images, captions) in enumerate(progress_bar):
        images = images.to(device)
        captions = list(map(list, zip(*captions)))
        
        loss = model(images, captions)
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(step+1):.4f}',
        })

        # Save at regular steps with current validation loss
        if (step + 1) % save_steps == 0 and scheduler is not None:
            curr_avg_loss = total_loss/(step+1)
            model.eval()  # Switch to eval mode for validation
            curr_val_loss = evaluate(model, val_loader)
            model.train()  # Switch back to training mode
            
            save_model(model, optimizer, curr_avg_loss, curr_val_loss, 
                      epoch, os.path.join('p2_checkpoints', f'epoch_{epoch}_step_{step+1}.pth'))
            print(f"\nStep {step+1} - Val Loss: {curr_val_loss:.4f}")
        
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    for images, captions in tqdm(val_loader, desc='Validating'):
        images = images.to(device)
        captions = list(map(list, zip(*captions)))
        
        loss = model(images, captions)
        total_loss += loss.item()
    
    return total_loss / num_batches

def save_model(model, optimizer, train_loss, val_loss, epoch, save_path):
    # Get only trainable parameters (LoRA and projection)
    trainable_state_dict = {
        k: v for k, v in model.state_dict().items() 
        if ("lora_" in k or "projection" in k)
    }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    torch.save(checkpoint, save_path)

def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    
    # Load only trainable parameters
    model_state_dict = model.state_dict()
    for k, v in checkpoint['model_state_dict'].items():
        if "lora_" in k or "projection" in k:
            model_state_dict[k] = v
    model.load_state_dict(model_state_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def main():
    model = ImageCaptioningModel().to(device)
    print("Initial model parameters:")
    print_params(model)
    
    # fixed json file so that every image has exactly 5 captions 
    train_dataset = ImageCaptionDataset(
        json_path='hw3_data/p2_data/train_fixed.json',
        img_dir='hw3_data/p2_data/images/train',
        tokenizer=model.tokenizer,
    )
    
    val_dataset = ImageCaptionDataset(
        json_path='hw3_data/p2_data/val_fixed.json',
        img_dir='hw3_data/p2_data/images/val',
        tokenizer=model.tokenizer,
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=12,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=40,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    num_epoch = 6
    num_warmup = 2
    lr = 1e-3

    os.makedirs('p2_checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Stage 1: Projection only training
    print("\nStage 1: Projection Training")
    optimizer = torch.optim.Adam([p for p in model.projection.parameters()], lr=lr)

    for epoch in range(1, num_warmup + 1):
        print(f"\nTraining Epoch {epoch}/{num_epoch} (Projection Only)")
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, val_loader, save_steps=104)
        val_loss = evaluate(model, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    # Stage 2: Joint training
    print("\nStage 2: Joint Training")
    all_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,  
        T_0=834,
        T_mult=2,
        eta_min=1e-4
    )

    for epoch in range(num_warmup + 1, num_epoch + 1):
        print(f"\nTraining Epoch {epoch}/{num_epoch} (Joint Training)")
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, val_loader, scheduler, save_steps=104)
        val_loss = evaluate(model, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    
if __name__ == "__main__":
    main()
