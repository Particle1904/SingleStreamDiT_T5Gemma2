import os
import torch
import math
import time
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
from tqdm import tqdm
from config import Config
from model_loader import load_vae
from text_encoder import TextEncoderWrapper

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_dense_buckets(target_res, stride=32):
    target_area = target_res * target_res
    buckets = set()
    
    for i in range(1, 41):
        for j in range(1, 41):
            ar = i / j
            if 0.25 <= ar <= 4.0:
                w = math.sqrt(target_area * ar)
                h = math.sqrt(target_area / ar)
                
                w = max(stride, round(w / stride) * stride)
                h = max(stride, round(h / stride) * stride)
                
                buckets.add((w, h))
                
    return sorted(list(buckets), key=lambda x: x[0]*x[1], reverse=True)

BUCKETS = generate_dense_buckets(Config.target_resolution, Config.bucket_alignment)

def get_best_bucket(w, h):
    target_aspect = w / h
    best_bucket = min(BUCKETS, key=lambda b: abs((b[0]/b[1]) - target_aspect))
    return best_bucket

def index_captions(dataset_dir):
    caption_index = {}
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.txt', '.caption')):
                base_name = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                
                if base_name not in caption_index:
                    caption_index[base_name] = []
                caption_index[base_name].append(full_path)
    return caption_index

def setup_models():
    print(f"Loading VAE: {Config.vae_id}...")
    vae = load_vae()
    
    print(f"Loading Text Encoder: {Config.text_model_id}...")
    text_encoder = TextEncoderWrapper(dtype=Config.dtype, device=Config.device)
    
    return vae, text_encoder

def process():
    os.makedirs(Config.cache_dir, exist_ok=True)
    BATCH_SIZE = Config.preprocess_batch_size
    
    print("Indexing all caption files recursively across all subfolders...")
    caption_index = index_captions(Config.dataset_dir)
    print(f"Indexed captions for {len(caption_index)} unique image keys.")
    
    all_files = [f for f in os.listdir(Config.dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    files_to_process = []
    skipped_count = 0
    
    for filename in all_files:
        base_name = os.path.splitext(filename)[0]
        cache_path = os.path.join(Config.cache_dir, f"{base_name}.pt")
        
        if os.path.exists(cache_path):
            skipped_count += 1
            continue
            
        if base_name not in caption_index or len(caption_index[base_name]) == 0:
            continue
            
        files_to_process.append(filename)
        
    print("\n" + "="*40)
    print("         CACHE RESUME STATUS")
    print("="*40)
    print(f"Total images found in dataset : {len(all_files)}")
    print(f"Already processed (Skipped)   : {skipped_count}")
    print(f"Remaining to process          : {len(files_to_process)}")
    print("="*40 + "\n")
    
    if len(files_to_process) == 0:
        print("All files are already processed! No work to do. Exiting safely.")
        return

    vae, text_encoder = setup_models()
    
    print("Starting preprocessing for remaining files with Batching & Dense Bucketing...")
    bucket_groups = {}
    bucket_counts = {}

    for filename in files_to_process:
        img_path = os.path.join(Config.dataset_dir, filename)
        base_name = os.path.splitext(filename)[0]

        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        bw, bh = get_best_bucket(img_w, img_h)
        
        res_key = (bw, bh)
        if res_key not in bucket_groups: 
            bucket_groups[res_key] = []
            
        bucket_groups[res_key].append(filename)
        bucket_counts[res_key] = bucket_counts.get(res_key, 0) + 1

    for res_key, filenames in bucket_groups.items():
        bw, bh = res_key
        print(f"Processing bucket {bw}x{bh} ({len(filenames)} images)...")
        
        for i in range(0, len(filenames), BATCH_SIZE):
            batch_files = filenames[i : i + BATCH_SIZE]
            img_tensors = []
            valid_batch_data = [] 

            for filename in batch_files:
                try:
                    img = Image.open(os.path.join(Config.dataset_dir, filename)).convert("RGB")
                    img_w, img_h = img.size
                    img_aspect = img_w / img_h
                    target_aspect = bw / bh
                    resize_w, resize_h = (int(bh * img_aspect), bh) if img_aspect > target_aspect else (bw, int(bw / img_aspect))
                    img = img.resize((resize_w, resize_h), resample=Image.LANCZOS)
                    left, top = (resize_w - bw) // 2, (resize_h - bh) // 2
                    img = img.crop((left, top, left + bw, top + bh))
                    
                    img_tensor = TF.to_tensor(img)
                    img_tensors.append(img_tensor)

                    base_name = os.path.splitext(filename)[0]
                    prompts = []
                    
                    cap_paths = caption_index.get(base_name, [])
                    for cap_path in cap_paths:
                        with open(cap_path, 'r', encoding='utf-8') as f:
                            prompt_text = f.read().strip()
                            if prompt_text:
                                prompts.append(prompt_text)

                    if len(prompts) == 0:
                        raise FileNotFoundError(f"No valid caption content found for {filename}")

                    with torch.no_grad():
                        text_embeds, attention_mask = text_encoder.encode(prompts)
                        
                    valid_batch_data.append({
                        "filename": filename,
                        "embeds": text_embeds.cpu().to(dtype=Config.dtype),
                        "mask": attention_mask.cpu().bool()
                    })
                except Exception as e:
                    print(f"Error prepping {filename}: {e}")

            if not img_tensors: 
                continue

            batch_tensor = torch.stack(img_tensors).to(Config.device)
            batch_tensor = TF.normalize(batch_tensor, [0.5], [0.5])
            
            with torch.no_grad(), torch.autocast(device_type=Config.device.type if isinstance(Config.device, torch.device) else torch.device(Config.device).type, dtype=Config.dtype):
                encoded = vae.encode(batch_tensor)
                latents_batch = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded[0]

            for idx, data in enumerate(valid_batch_data):
                active_indices = data["mask"].nonzero()
                if active_indices.numel() > 0:
                    max_len = int(active_indices[:, 1].max().item() + 1)
                else:
                    max_len = 1
                
                save_data = {
                    "latents": latents_batch[idx].cpu().to(dtype=Config.dtype),
                    "text_embeds_list": data["embeds"][:, :max_len].clone(),
                    "attention_mask_list": data["mask"][:, :max_len].clone(),
                    "width": bw,
                    "height": bh
                }
                torch.save(save_data, os.path.join(Config.cache_dir, f"{os.path.splitext(data['filename'])[0]}.pt"))

    print("\n" + "="*30)
    print(f"      STATS FOR {Config.target_resolution}px      ")
    print("="*30)
    total_images = sum(bucket_counts.values())
    for res_key in sorted(bucket_counts.keys(), key=lambda x: x[0]*x[1], reverse=True):
        print(f"[{res_key[0]}x{res_key[1]}]: {bucket_counts[res_key]:3d} images ({(bucket_counts[res_key]/total_images)*100:.1f}%)")
    print("="*30)
    
if __name__ == "__main__":
    start_time = time.time()
    process()
    print(f"Total time in minutes: {(time.time() - start_time) / 60:.2f}")