import torch
import os
from tqdm import tqdm
from config import Config

CACHE_DIR = Config.cache_dir

def calculate_stats():
    files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.pt')]
    if not files:
        print("No files found!")
        return

    print("Calculating Latent Statistics (Welford's Algorithm)...")
    
    n_samples = 0
    mean_accumulator = 0.0
    M2_accumulator = 0.0
    
    for f in tqdm(files):
        data = torch.load(f, map_location="cpu")
        l = data["latents"].float() 
        
        pixels = l.numel()
        batch_mean = l.mean().item()
        
        delta = batch_mean - mean_accumulator
        n_samples += 1
        mean_accumulator += delta / n_samples
        M2_accumulator += delta * (batch_mean - mean_accumulator)

    all_means = []
    all_stds = []
    
    for f in tqdm(files):
        data = torch.load(f, map_location="cpu")
        l = data["latents"].float()
        all_means.append(l.mean().item())
        all_stds.append(l.std().item())
        
    total_mean = sum(all_means) / len(all_means)
    total_std = sum(all_stds) / len(all_stds)

    print(f"\n" + "="*40)
    print(f"      RESULTS TO COPY TO CONFIG.PY      ")
    print(f"="*40)
    print(f"Current vae_scaling_factor used: {Config.vae_scaling_factor}")
    print(f"dataset_mean = {total_mean:.6f}")
    print(f"dataset_std  = {total_std:.6f}")
    print(f"="*40)
    
    if total_std < 0.5:
        print("\n[!] WARNING: Your std is very low. You MUST update config.py with these values.")

if __name__ == "__main__":
    calculate_stats()