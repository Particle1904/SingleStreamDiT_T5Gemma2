import torch
import glob
import os
import random
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from config import Config
from latents import normalize_latents

# Used for Validation on 200 flowers dataset.
def split_dataset_indices(total_files, items_per_category=20, val_per_category=4):
    train_indices = []
    val_indices = []
    for i in range(total_files):
        # 0..19
        pos = i % items_per_category
        # if pos is 16, 17, 18, 19 -> Val
        if pos >= (items_per_category - val_per_category):
            val_indices.append(i)
        else:
            train_indices.append(i)
    return set(train_indices), set(val_indices)

class TextImageDataset(Dataset):
    def __init__(self):
        self.files = glob.glob(os.path.join(Config.cache_dir, "*.pt"))
        self.files.sort() 
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {Config.cache_dir}")
        
        self.cache = {}        
        self.buckets = {}
        
        print(f"Indexing {len(self.files)} files (RAM Cache: {Config.load_entire_dataset})...")
        for idx, f in enumerate(tqdm(self.files, desc="Loading Dataset", disable=not Config.accelerator.is_main_process)):
            try:
                d = torch.load(f, map_location="cpu")
                
                if Config.load_entire_dataset:
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
        if Config.load_entire_dataset:
            data = self.cache[idx]
        else:
            data = torch.load(self.files[idx], map_location="cpu")

        latents = data["latents"]
        if "text_embeds_list" in data:
            num_captions = data["text_embeds_list"].shape[0]
            choice_idx = random.randint(0, num_captions - 1)
            text = data["text_embeds_list"][choice_idx]
            mask = data["attention_mask_list"][choice_idx]
        else:
            text = data["text_embeds"]
            mask = data.get("attention_mask", torch.ones(text.shape[0], dtype=torch.bool))

        target_len = Config.max_token_length
        if text.shape[0] > target_len:
            text = text[:target_len]
            mask = mask[:target_len]

        latents = normalize_latents(latents)
        if Config.flip_aug and random.random() < 0.5:
            latents = torch.flip(latents, dims=[-1])
            
        return {
            "latents": latents.to(Config.dtype),
            "text_embeds": text.to(Config.dtype),
            "text_mask": mask,
            "height": data["height"],
            "width": data["width"]
        }
        
def dynamic_collate_fn(batch):
    # 1. Fast path for fixed-size properties
    latents = torch.stack([item["latents"] for item in batch])
    heights = torch.tensor([item["height"] for item in batch])
    widths = torch.tensor([item["width"] for item in batch])
    
    # 2. Dynamic Text Padding (Pre-allocation method for maximum speed)
    texts = [item["text_embeds"] for item in batch]
    masks = [item["text_mask"] for item in batch]
    
    # Find the maximum sequence length in THIS specific batch
    max_len = max(t.shape[0] for t in texts)
    B = len(batch)
    D = texts[0].shape[1] # Text embedding dimension
    dtype = texts[0].dtype
    
    # Pre-allocate contiguous blocks of memory (blazingly fast)
    batched_texts = torch.zeros((B, max_len, D), dtype=dtype)
    batched_masks = torch.zeros((B, max_len), dtype=torch.bool)
    
    # Copy data into the pre-allocated tensors
    for i in range(B):
        seq_len = texts[i].shape[0]
        batched_texts[i, :seq_len, :] = texts[i]
        batched_masks[i, :seq_len] = masks[i]
        
    return {
        "latents": latents,
        "text_embeds": batched_texts,
        "text_mask": batched_masks,
        "height": heights,
        "width": widths
    }

class BucketBatchSampler(Sampler):
    def __init__(self, buckets, batch_size, drop_last=False):
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