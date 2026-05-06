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

def generate_buckets(target_res, stride=32):
    area = target_res * target_res
    aspect_ratios = [
        1.0, 1.333, 0.75, 1.5, 0.666, 1.777, 0.562, 2.0, 0.5
    ]
    if target_res <= 512:
        aspect_ratios = [1.0, 1.333, 0.75]

    buckets = set()
    for ar in aspect_ratios:
        w = math.sqrt(area * ar)
        h = math.sqrt(area / ar)
        w = round(w / stride) * stride
        h = round(h / stride) * stride
        buckets.add((w, h))
    return sorted(list(buckets), key=lambda x: x[0]*x[1], reverse=True)

BUCKETS = generate_buckets(Config.target_resolution, Config.bucket_alignment)

def get_best_bucket(w, h):
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
    
    files_in_dir = os.listdir(Config.cache_dir)
    if len(files_in_dir) > 0:
        print(f"Overwriting cache in {Config.cache_dir}...")
        for f in files_in_dir:
            if f.endswith('.pt'):
                os.remove(os.path.join(Config.cache_dir, f))
                
    vae, tokenizer, text_model = setup_models()
    print("Starting preprocessing with Strict Upscale Prevention...")
    
    files = [f for f in os.listdir(Config.dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    bucket_counts = {}
    
    for filename in tqdm(files):
        try:
            img_path = os.path.join(Config.dataset_dir, filename)
            txt_path = os.path.join(Config.dataset_dir, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(txt_path):
                 txt_path = os.path.join(Config.dataset_dir, os.path.splitext(filename)[0] + ".caption")
                 if not os.path.exists(txt_path): continue

            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
            
            bw, bh = get_best_bucket(img_w, img_h)
            
            if Config.dynamic_buckets:
                bw = min(bw, (img_w // stride) * stride)
                bh = min(bh, (img_h // stride) * stride)
            
            bw = max(stride, bw)
            bh = max(stride, bh)
            
            if (bw, bh) not in bucket_counts: bucket_counts[(bw, bh)] = 0
            bucket_counts[(bw, bh)] += 1
            
            target_aspect = bw / bh
            img_aspect = img_w / img_h

            if img_aspect > target_aspect:
                resize_h = bh
                resize_w = int(bh * img_aspect)
            else:
                resize_w = bw
                resize_h = int(bw / img_aspect)
            
            img = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
            left = (resize_w - bw) // 2
            top = (resize_h - bh) // 2
            img = img.crop((left, top, left + bw, top + bh))
            
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(Config.device)
            img_tensor = TF.normalize(img_tensor, [0.5], [0.5])

            with torch.no_grad():
                encoded = vae.encode(img_tensor)
                # Handle standard AutoencoderKL (probabilistic) vs AutoencoderDC (deterministic)
                if hasattr(encoded, "latent_dist"):
                    latents = encoded.latent_dist.mode()
                elif hasattr(encoded, "latents"):
                    latents = encoded.latents
                else:
                    latents = encoded[0]

            base_name = os.path.splitext(filename)[0]
            
            caption_files =[]
            main_txt = os.path.join(Config.dataset_dir, f"{base_name}.txt")
            if os.path.exists(main_txt):
                caption_files.append(main_txt)
                
            for item in os.listdir(Config.dataset_dir):
                sub_dir = os.path.join(Config.dataset_dir, item)
                if os.path.isdir(sub_dir):
                    sub_txt = os.path.join(sub_dir, f"{base_name}.txt")
                    if os.path.exists(sub_txt):
                        caption_files.append(sub_txt)
            
            if not caption_files:
                print(f"Warning: No captions found for {filename}, skipping.")
                continue

            embeds_list = []
            masks_list =[]
            
            for cap_path in caption_files:
                with open(cap_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                
                inputs = tokenizer(prompt, max_length=Config.max_token_length, padding="max_length", truncation=True, return_tensors="pt").to(Config.device)
                text_attention_mask = inputs.attention_mask.squeeze(0).cpu().bool()
                
                with torch.no_grad():
                    outputs = text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                    text_embeds = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                    
                embeds_list.append(text_embeds.squeeze(0).cpu().to(dtype=Config.dtype))
                masks_list.append(text_attention_mask)
            
            stacked_embeds = torch.stack(embeds_list)
            stacked_masks = torch.stack(masks_list)

            save_data = {
                "latents": latents.squeeze(0).cpu().to(dtype=Config.dtype),
                "text_embeds_list": stacked_embeds,
                "attention_mask_list": stacked_masks,
                "width": bw,
                "height": bh
            }
            torch.save(save_data, os.path.join(Config.cache_dir, os.path.splitext(filename)[0] + ".pt"))

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        
    print("\n" + "="*30)
    print(f"      STATS FOR {Config.target_resolution}px      ")
    print("="*30)
    total_images = sum(bucket_counts.values())
    sorted_buckets = sorted(bucket_counts.keys(), key=lambda x: x[0]*x[1], reverse=True)
    for res_key in sorted_buckets:
        count = bucket_counts[res_key]
        percentage = (count / total_images) * 100
        print(f"[{res_key[0]}x{res_key[1]}]: {count:3d} images ({percentage:.1f}%)")
    print("="*30)

if __name__ == "__main__":
    start_time = time.time()
    process()
    print(f"Total time in minutes: {(time.time() - start_time) / 60:.2f}")