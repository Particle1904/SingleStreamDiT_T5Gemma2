import torch
import sys
import os
from diffusers import AutoencoderKL
from PIL import Image
# Import config and latents from the folder above.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config
from latents import *

TARGET_FILE = Config.target_file
VAE_ID = Config.vae_id

def check():
    print(f"Checking {TARGET_FILE}...")
    
    data = torch.load(TARGET_FILE)
    latents = data["latents"].unsqueeze(0).to("cuda")

    print(f"Loading VAE: {VAE_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to("cuda")
    
    print("Decoding...")
    latents = to_vae_space(latents)
    
    with torch.no_grad():
        image = vae.decode(latents.float()).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    
    Image.fromarray(image[0]).save("cache_verification.png")
    print("Saved cache_verification.png. Check this image!")

if __name__ == "__main__":
    check()