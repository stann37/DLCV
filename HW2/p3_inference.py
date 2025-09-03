import sys
import os
import torch
import json
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor
import gc

# Add the stable-diffusion directory to the Python path
sys.path.append("stable-diffusion")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model.eval()
    return model

def save_image(data, path):
    Image.fromarray((data * 255).astype(np.uint8)).save(path)

@torch.inference_mode()
def process_batch(model, sampler, prompts, guidance_scale, batch_size):
    with model.ema_scope():
        uc = model.get_learned_conditioning([""] * batch_size)
        c = model.get_learned_conditioning(prompts)
        shape = [4, 64, 64]
        samples, _ = sampler.sample(S=50,
                                    conditioning=c,
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=guidance_scale,
                                    unconditional_conditioning=uc,
                                    eta=0.0)
        
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

    return x_samples

def inference(config_path, ckpt_path, json_path, output_dir, embeddings_dir, batch_size=4):
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)

    torch.cuda.empty_cache()
    gc.collect()

    sampler = DPMSolverSampler(model)

    with open(json_path, 'r') as f:
        data = json.load(f)

    token_settings = {
        "<new1>": {"guidance_scale": 7.5, "num_vectors": 4},
        "<new2>": {"guidance_scale": 8.5, "num_vectors": 3}
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        for concept_id, concept_data in data.items():
            token_name = concept_data['token_name']
            prompts = concept_data['prompt']
            
            settings = token_settings.get(token_name, {"guidance_scale": 10, "num_vectors": 3})
            guidance_scale = settings["guidance_scale"]
            num_vectors = settings["num_vectors"]
            
            # Load and move embeddings to CUDA
            embeddings_path = os.path.join(embeddings_dir, f"{token_name[1:-1]}_final.pt")
            learned_embeds = torch.load(embeddings_path).cuda()  # Move to CUDA immediately
            
            placeholder_tokens = [f"{token_name[1:-1]}-{i}" for i in range(num_vectors)]
            num_added_tokens = model.cond_stage_model.tokenizer.add_tokens(placeholder_tokens)
            model.cond_stage_model.transformer.resize_token_embeddings(len(model.cond_stage_model.tokenizer))
            
            token_ids = [model.cond_stage_model.tokenizer.convert_tokens_to_ids(token) for token in placeholder_tokens]
            model.cond_stage_model.transformer.get_input_embeddings().weight.data[token_ids] = learned_embeds

            concept_dir = os.path.join(output_dir, concept_id)
            os.makedirs(concept_dir, exist_ok=True)
            
            for prompt_idx, prompt in enumerate(prompts):
                prompt = prompt.replace(token_name, " ".join(placeholder_tokens))
                prompt_dir = os.path.join(concept_dir, str(prompt_idx))
                os.makedirs(prompt_dir, exist_ok=True)
                
                batch_prompts = [prompt] * batch_size
                futures = []
                
                for j in tqdm(range(0, 25, batch_size), desc="Generating images"):
                    try:
                        x_samples = process_batch(model, sampler, batch_prompts, guidance_scale, batch_size)
                        
                        for k, x_sample in enumerate(x_samples):
                            if j + k < 25:
                                save_path = os.path.join(prompt_dir, f"source{concept_id}_prompt{prompt_idx}_{j+k}.png")
                                futures.append(executor.submit(save_image, x_sample, save_path))
                                
                    except RuntimeError as e:
                        print(f"Error processing prompt: {e}")
                        print(f"Skipping batch at index {j}")
                        continue
                
                for future in futures:
                    future.result()

                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    config_path = "stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    json_path = sys.argv[1]
    output_dir = sys.argv[2]
    embeddings_dir = "p3_embeddings_best"
    ckpt_path = sys.argv[3]
    batch_size = 3

    inference(config_path, ckpt_path, json_path, output_dir, embeddings_dir, batch_size)