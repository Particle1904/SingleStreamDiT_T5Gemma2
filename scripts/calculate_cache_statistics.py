import torch
import os
import sys
from tqdm import tqdm
# Import config from the folder above.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config

CACHE_DIR = Config.cache_dir

def calculate_stats():
    files =[os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.pt')]
    if not files:
        print("No files found!")
        return

    print("Calculating Exact Global Latent Statistics...")
    n_elements = 0
    sum_x = 0.0
    sum_sq_x = 0.0

    for f in tqdm(files):
        data = torch.load(f, map_location="cpu")
        l = data["latents"].to(torch.float64) 
        
        n_elements += l.numel()
        sum_x += l.sum().item()
        sum_sq_x += (l ** 2).sum().item()

    total_mean = sum_x / n_elements
    variance = (sum_sq_x / n_elements) - (total_mean ** 2)
    total_std = variance ** 0.5

    print(f"\n" + "="*40)
    print(f"      RESULTS TO COPY TO CONFIG.PY      ")
    print(f"="*40)
    print(f"dataset_mean = {total_mean:.6f}")
    print(f"dataset_std  = {total_std:.6f}")
    print(f"="*40)
    
if __name__ == "__main__":
    calculate_stats()