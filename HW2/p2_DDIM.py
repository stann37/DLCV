import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

class DDIM:
    def __init__(self, nn_model, timesteps=1000, beta_schedule=beta_scheduler(), device='cuda', seed=42):
        self.model = nn_model.to(device)
        self.timesteps = timesteps
        self.betas = beta_schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.device = device

        # Set the random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    @torch.no_grad()
    def sample(self, noise, ddim_timesteps=50, ddim_eta=0.0):
        noise = noise.to(self.device)
        batch_size = noise.size(0)
        sample_img = noise
        
        #ddim_timestep_seq = np.linspace(0, self.timesteps - 1, ddim_timesteps, dtype=int)
        ddim_timestep_seq = np.arange(1, 982, 20, dtype=int)

        #print(ddim_timestep_seq)
        
        for i in tqdm(reversed(range(ddim_timesteps)), desc="Sampling loop time step"):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=self.device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_seq[max(0, i - 1)], device=self.device, dtype=torch.long)
            
            #t = t - 1
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
            pred_noise = self.model(sample_img, t)
            
            pred_x0 = (sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            sigmas_t = ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)
            sample_img = x_prev
        
        return sample_img.cpu()
