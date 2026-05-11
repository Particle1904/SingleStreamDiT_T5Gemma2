import os
import torch
import math
import time
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
from tqdm import tqdm
from config import Config
from model_loader import load_vae

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
    # Match by aspect ratio proximity
    target_aspect = w / h
    best_bucket = min(BUCKETS, key=lambda b: abs((b[0]/b[1]) - target_aspect))
    return best_bucket

def setup_models():
    print(f"Loading VAE: {Config.vae_id}...")
    vae = load_vae()
    print(f"Loading Text Encoder: {Config.text_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.text_model_id)
    full_model = AutoModel.from_pretrained(Config.text_model_id, trust_remote_code=True)
    text_model = full_model.encoder if hasattr(full_model, "encoder") else full_model
    text_model.to(Config.device, dtype=Config.dtype).eval()
    return vae, tokenizer, text_model

def process():
    os.makedirs(Config.cache_dir, exist_ok=True)
    stride = Config.bucket_alignment
    BATCH_SIZE = Config.preprocess_batch_size
    
    files_in_dir = os.listdir(Config.cache_dir)
    if len(files_in_dir) > 0:
        print(f"Overwriting cache in {Config.cache_dir}...")
        for f in files_in_dir:
            if f.endswith('.pt'):
                os.remove(os.path.join(Config.cache_dir, f))
                
    vae, tokenizer, text_model = setup_models()
    print("Starting preprocessing with Batching & Dense Bucketing...")
    
    files =[f for f in os.listdir(Config.dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    bucket_groups = {}
    bucket_counts = {}

    # 1. First loop: Just categorize images into buckets by their dimensions
    for filename in files:
        img_path = os.path.join(Config.dataset_dir, filename)
        base_name = os.path.splitext(filename)[0]
        if not (os.path.exists(os.path.join(Config.dataset_dir, f"{base_name}.txt")) or 
                os.path.exists(os.path.join(Config.dataset_dir, f"{base_name}.caption"))):
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        bw, bh = get_best_bucket(img_w, img_h)
        
        res_key = (bw, bh)
        if res_key not in bucket_groups: 
            bucket_groups[res_key] =[]
            
        bucket_groups[res_key].append(filename)
        bucket_counts[res_key] = bucket_counts.get(res_key, 0) + 1

    # 2. Second loop: Process each bucket in batches
    for res_key, filenames in bucket_groups.items():
        bw, bh = res_key
        print(f"Processing bucket {bw}x{bh} ({len(filenames)} images)...")
        
        for i in range(0, len(filenames), BATCH_SIZE):
            batch_files = filenames[i : i + BATCH_SIZE]
            img_tensors =[]
            valid_batch_data =[] 

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
                    cap_path = os.path.join(Config.dataset_dir, f"{base_name}.txt")
                    if not os.path.exists(cap_path): 
                        cap_path = os.path.join(Config.dataset_dir, f"{base_name}.caption")
                    
                    with open(cap_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    inputs = tokenizer(prompt, max_length=Config.max_token_length, padding="max_length", truncation=True, return_tensors="pt").to(Config.device)
                    with torch.no_grad():
                        outputs = text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                        text_embeds = (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]).squeeze(0)
                    
                    valid_batch_data.append({
                        "filename": filename,
                        "embeds": text_embeds.cpu().to(dtype=Config.dtype),
                        "mask": inputs.attention_mask.squeeze(0).cpu().bool()
                    })
                except Exception as e:
                    print(f"Error prepping {filename}: {e}")

            if not img_tensors: 
                continue

            # This is exactly where this logic belongs!
            batch_tensor = torch.stack(img_tensors).to(Config.device)
            batch_tensor = TF.normalize(batch_tensor,[0.5], [0.5])
            
            with torch.no_grad(), torch.autocast(device_type=Config.device.type if isinstance(Config.device, torch.device) else torch.device(Config.device).type, dtype=Config.dtype):
                encoded = vae.encode(batch_tensor)
                latents_batch = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded[0]

            for idx, data in enumerate(valid_batch_data):
                save_data = {
                    "latents": latents_batch[idx].cpu().to(dtype=Config.dtype),
                    "text_embeds_list": data["embeds"].unsqueeze(0),
                    "attention_mask_list": data["mask"].unsqueeze(0),
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