import os
import sys
import torch
import shutil
import random
import glob
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)

from config import Config

def create_fixed_resolution_subset(subset_size=32):
    source_dir = Config.cache_dir
    subset_dir = os.path.join(Config.output_dir, f"subset_{subset_size}")
    os.makedirs(subset_dir, exist_ok=True)

    print(f"Scanning cache directory: {source_dir}")
    pt_files = glob.glob(os.path.join(source_dir, "*.pt"))
    
    if not pt_files:
        print(f"Error: No .pt files found in {source_dir}")
        return

    print(f"Found {len(pt_files)} cached files. Reading resolution statistics...")
    buckets = defaultdict(list)

    for f in tqdm(pt_files, desc="Grouping by resolution"):
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            w = data.get("width")
            h = data.get("height")
            
            if w is not None and h is not None:
                buckets[(w, h)].append(f)
            else:
                print(f"Warning: {os.path.basename(f)} is missing width/height metadata. Skipping.")
        except Exception as e:
            print(f"Failed to read {os.path.basename(f)}: {e}")

    sorted_buckets = sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("\n" + "="*40)
    print("      DATASET RESOLUTION STATS")
    print("="*40)
    for (w, h), files in sorted_buckets:
        print(f"[{w}x{h}]: {len(files)} images")
    print("="*40 + "\n")

    valid_buckets = [(res, files) for res, files in sorted_buckets if len(files) >= subset_size]

    if not valid_buckets:
        print(f"Error: No single resolution has at least {subset_size} images. Cannot create a fixed-resolution subset.")
        return

    best_res, best_files = valid_buckets[0]
    print(f"Selected resolution {best_res[0]}x{best_res[1]} (contains {len(best_files)} images).")

    random.seed(42)
    selected_files = random.sample(best_files, subset_size)

    print(f"Copying {subset_size} files to {subset_dir}...")
    for f in tqdm(selected_files, desc="Copying files"):
        target_path = os.path.join(subset_dir, os.path.basename(f))
        shutil.copy2(f, target_path)

    print(f"\nSuccess! Created subset at: {subset_dir}")

if __name__ == "__main__":
    create_fixed_resolution_subset(subset_size=32)