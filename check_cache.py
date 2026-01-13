import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
from config import Config

TARGET_FILE = Config.target_file
VAE_ID = Config.vae_id

def check():
    print(f"Checking {TARGET_FILE}...")
    
    data = torch.load(TARGET_FILE)
    latents = data["latents"].unsqueeze(0).to("cuda")

    print(f"Loading VAE: {VAE_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to("cuda")
    
    print("Decoding...")
    latents = latents / Config.vae_scaling_factor
    
    with torch.no_grad():
        image = vae.decode(latents.float()).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    
    Image.fromarray(image[0]).save("cache_verification.png")
    print("Saved cache_verification.png. Check this image!")

if __name__ == "__main__":
    check()