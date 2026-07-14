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

def create_proportional_subset(subset_size=1024):
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

    print("\n" + "=" * 40)
    print("      DATASET RESOLUTION STATS")
    print("=" * 40)
    for (w, h), files in sorted_buckets:
        print(f"[{w}x{h}]: {len(files)} images")
    print("=" * 40 + "\n")

    all_files = [f for files in buckets.values() for f in files]
    if len(all_files) < subset_size:
        print(f"Error: only {len(all_files)} total files available, cannot draw {subset_size}.")
        return

    random.seed(42)
    selected_files = random.sample(all_files, subset_size)

    selected_buckets = defaultdict(int)
    for f in selected_files:
        for res, files in buckets.items():
            if f in files:
                selected_buckets[res] += 1
                break
    print("Subset bucket distribution (top 10):")
    for res, count in sorted(selected_buckets.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  [{res[0]}x{res[1]}]: {count}")

    print(f"\nCopying {subset_size} files to {subset_dir}...")
    for f in tqdm(selected_files, desc="Copying files"):
        target_path = os.path.join(subset_dir, os.path.basename(f))
        shutil.copy2(f, target_path)

    print(f"\nSuccess! Created subset at: {subset_dir}")

if __name__ == "__main__":
    create_proportional_subset(subset_size=1024)