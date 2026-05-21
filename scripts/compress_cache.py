import os
import torch
import sys
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config

def compress():
    files = [os.path.join(Config.cache_dir, f) for f in os.listdir(Config.cache_dir) if f.endswith('.pt')]
    if not files:
        print("No cache files found.")
        return

    print(f"Compacting {len(files)} files on disk (Reclaiming physical LFS memory)...")
    reclaimed_bytes = 0

    for f_path in tqdm(files):
        data = torch.load(f_path, map_location="cpu")
        
        embeds = data.get("text_embeds_list", None)
        mask = data.get("attention_mask_list", None)
        
        if embeds is None or mask is None:
            continue
            
        try:
            old_storage_bytes = embeds.untyped_storage().nbytes()
        except AttributeError:
            old_storage_bytes = embeds.storage().nbytes()

        cloned_embeds = embeds.clone()
        cloned_mask = mask.clone()
        
        try:
            new_storage_bytes = cloned_embeds.untyped_storage().nbytes()
        except AttributeError:
            new_storage_bytes = cloned_mask.storage().nbytes()

        data["text_embeds_list"] = cloned_embeds
        data["attention_mask_list"] = cloned_mask
        
        reclaimed_bytes += (old_storage_bytes - new_storage_bytes)

        temp_path = f_path + ".tmp"
        torch.save(data, temp_path)
        os.replace(temp_path, f_path)

    print(f"\nCompression Complete!")
    print(f"Reclaimed Physical Disk Space: {reclaimed_bytes / (1024**3):.2f} GB of actual hard drive space.")

if __name__ == "__main__":
    compress()