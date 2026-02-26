import torch
import sys
import os
from PIL import Image

# Import config and latents from the folder above.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config
from latents import *
from model_loader import load_vae

def check():
    print(f"Checking {Config.target_file}...")
    
    data = torch.load(Config.target_file)
    latents = data["latents"].unsqueeze(0).to("cuda")

    print(f"Loading VAE: {Config.vae_id}...")
    vae = load_vae()
    
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