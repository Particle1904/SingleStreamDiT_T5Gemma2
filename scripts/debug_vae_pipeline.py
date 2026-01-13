import torch
import os
import sys
from diffusers import AutoencoderKL
from PIL import Image
# Import config and latents from the folder above.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config
from latents import denormalize_latents, to_vae_space

DEVICE = Config.device
DTYPE = Config.dtype
VAE_ID = Config.vae_id
TARGET_FILE = Config.target_file

def check_pipeline():
    print(f"--- DEBUGGING PIPELINE (Forced Float32 VAE) ---")
    print(f"Target File: {TARGET_FILE}")
    print(f"Config Mean: {Config.dataset_mean}")
    print(f"Config Std:  {Config.dataset_std}")
    print(f"VAE Scaling: {Config.vae_scaling_factor}")
    
    data = torch.load(TARGET_FILE, map_location=DEVICE)
    latents = data["latents"].to(torch.float32).unsqueeze(0) # [1, 16, H, W]
    
    latents = latents.float()
    
    print("Denormalizing (in float32)...")
    latents = denormalize_latents(latents) 
    
    print("Scaling to VAE space (in float32)...")
    latents = to_vae_space(latents)

    print("Loading VAE (Float32)...")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE, dtype=torch.float32).eval()
    
    print("Decoding...")
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    
    out_path = "debug_pipeline_float32.png"
    Image.fromarray(image[0]).save(out_path)
    print(f"Saved result to {out_path}")
    print("Check this image.")

if __name__ == "__main__":
    check_pipeline()