import torch
from config import Config
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
from transformers import AutoModel

def load_vae():
    vae = None
    if "FLUX.2" in Config.vae_id:
        print("Loading FLUX2 VAE...")
        vae = DiffusersAutoencoderKL.from_pretrained(Config.vae_id, low_cpu_mem_usage=True).to(Config.device, dtype=torch.float32).eval()
    elif "dc-ae" in Config.vae_id.lower():
        from diffusers import AutoencoderDC
        print("Loading DC-AE (Sana) VAE...")
        vae = AutoencoderDC.from_pretrained(Config.vae_id, low_cpu_mem_usage=True).to(Config.device, dtype=torch.float32).eval()
    else:
        print("Loading generic VAE...")
        vae = AutoencoderKL.from_pretrained(Config.vae_id, low_cpu_mem_usage=True).to(Config.device, dtype=torch.float32).eval()
    return vae

def load_dinov3():
    print(f"Loading DINOv3: {Config.repa_model}...")
    dino = AutoModel.from_pretrained(Config.repa_model).to(Config.device, dtype=Config.dtype).eval()
    for p in dino.parameters():
        p.requires_grad = False
    return dino