import torch
import sys
import os
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config
from latents import *
from model_loader import load_vae

def check():
    print(f"Checking {Config.target_file}...")
    
    data = torch.load(Config.target_file, map_location="cpu")

    print("\n" + "="*40)
    print("        TEXT EMBEDDING STATS")
    print("="*40)
    
    if "text_embeds_list" in data:
        embeds = data["text_embeds_list"]
        mask = data["attention_mask_list"]
        print("Format: Multi-Caption Cache")
    elif "text_embeds" in data:
        embeds = data["text_embeds"].unsqueeze(0)
        mask = data["attention_mask"].unsqueeze(0)
        print("Format: Single-Caption Cache")
    else:
        print("ERROR: No text embeddings found in cache!")
        embeds = None
        
    if embeds is not None:
        print(f"Total Captions Saved : {embeds.shape[0]}")
        print(f"Embeddings Shape     : {embeds.shape} (Captions, Seq_Len, Hidden_Size)")
        print(f"Mask Shape           : {mask.shape}")
        
        print("\n--- Tensor Health ---")
        print(f"Min Value  : {embeds.min().item():.4f}")
        print(f"Max Value  : {embeds.max().item():.4f}")
        print(f"Mean Value : {embeds.mean().item():.4f}")
        print(f"Has NaNs?  : {torch.isnan(embeds).any().item()}")
        
        print("\n--- Token Usage (per caption) ---")
        active_tokens = mask.sum(dim=1).tolist()
        for i, count in enumerate(active_tokens):
            print(f"Caption {i+1}: {count} active tokens (out of {mask.shape[1]})")
                    
    print("\n" + "="*40)
    print("          IMAGE LATENTS")
    print("="*40)
    
    latents = data["latents"].unsqueeze(0).to("cuda")
    print(f"Target Resolution : {data.get('width')}x{data.get('height')}")
    print(f"Latent Shape      : {latents.shape}")

    print(f"\nLoading VAE: {Config.vae_id}...")
    vae = load_vae()
    
    print("Decoding latents to RGB space...")
    latents = to_vae_space(latents)
    
    with torch.no_grad():
        image = vae.decode(latents.float()).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    
    save_path = "cache_verification.png"
    Image.fromarray(image[0]).save(save_path)
    print(f"Success! Saved image to: {save_path}")

if __name__ == "__main__":
    check()