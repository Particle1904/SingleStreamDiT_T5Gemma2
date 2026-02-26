import torch
from config import Config
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL

def load_vae():
    vae = None  
    if "FLUX.2" in Config.vae_id:
        print("Loading FLUX2 VAE...")
        vae = DiffusersAutoencoderKL.from_pretrained(Config.vae_id, low_cpu_mem_usage=True).to(Config.device, dtype=torch.float32).eval()
    else:
        print("Loading generic VAE...")
        vae = AutoencoderKL.from_pretrained(Config.vae_id, low_cpu_mem_usage=True).to(Config.device, dtype=torch.float32).eval()
        
    return vae