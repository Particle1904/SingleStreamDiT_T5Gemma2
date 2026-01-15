import torch
import glob
import os
import random
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from config import Config
from latents import normalize_latents

CACHE_DIR = Config.cache_dir
LOAD_ENTIRE_DATASET = Config.load_entire_dataset

FLIP_AUG = Config.flip_aug
TEXT_DROPOUT = Config.text_dropout

class TextImageDataset(Dataset):
    def __init__(self):
        self.files = glob.glob(os.path.join(CACHE_DIR, "*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {CACHE_DIR}")
        
        self.cache = {}
        
        self.buckets = {}
        
        print(f"Indexing {len(self.files)} files (RAM Cache: {LOAD_ENTIRE_DATASET})...")
        for idx, f in enumerate(tqdm(self.files, desc="Loading Dataset", disable=not Config.accelerator.is_main_process)):
            try:
                d = torch.load(f, map_location="cpu")
                
                if LOAD_ENTIRE_DATASET:
                    self.cache[idx] = d
                
                res_key = (d["height"], d["width"])
                if res_key not in self.buckets:
                    self.buckets[res_key] = []
                self.buckets[res_key].append(idx)
                
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if LOAD_ENTIRE_DATASET:
            data = self.cache[idx]
        else:
            data = torch.load(self.files[idx], map_location="cpu")

        latents = data["latents"]
        text = data["text_embeds"]
        
        latents = normalize_latents(latents)
        
        if FLIP_AUG and random.random() < 0.5:
            latents = torch.flip(latents, dims=[-1])
            
        if random.random() < TEXT_DROPOUT:
            text = torch.zeros_like(text)
            
        return {
            "latents": latents,
            "text_embeds": text,
            "height": data["height"],
            "width": data["width"]
        }

class BucketBatchSampler(Sampler):
    def __init__(self, buckets, batch_size, drop_last=True):
        self.buckets = buckets
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        
        for res_key, indices in self.buckets.items():
            random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                
                batches.append(batch)
        
        random.shuffle(batches)
        
        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total