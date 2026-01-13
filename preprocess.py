import os
import torch
import math
import time
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL
from tqdm import tqdm
from config import Config

TARGET_RESOLUTION = Config.target_resolution
MAX_TOKEN_LENGTH = Config.max_token_length

DATASET_DIR = Config.dataset_dir
OUTPUT_DIR = Config.cache_dir
TEXT_MODEL_ID = Config.text_model_id
VAE_MODEL_ID = Config.vae_id
VAE_SCALING_FACTOR = Config.vae_scaling_factor
BUCKET_ALIGNMENT = Config.bucket_alignment

DEVICE = Config.device
DTYPE = Config.dtype

def generate_buckets(target_res, stride=32):
    area = target_res * target_res
    
    aspect_ratios = [
        1.0,           # Square
        1.333, 0.75,   # 4:3 & 3:4
        1.5,   0.666,  # 3:2 & 2:3
        1.777, 0.562,  # 16:9 & 9:16
        2.0,   0.5     # 2:1 & 1:2 (Ultrawide/Tall)
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

BUCKETS = generate_buckets(TARGET_RESOLUTION, BUCKET_ALIGNMENT)

print(f"--- CONFIGURATION ---")
print(f"Target Resolution: {TARGET_RESOLUTION}x{TARGET_RESOLUTION}")
print(f"Output Directory:  {OUTPUT_DIR}")
print(f"Generated {len(BUCKETS)} Buckets:")

for b in BUCKETS:
    print(f" - {b[0]} x {b[1]} (AR: {b[0]/b[1]:.2f})")
print("---------------------")

def get_best_bucket(w, h):
    target_aspect = w / h
    best_bucket = min(BUCKETS, key=lambda b: abs((b[0]/b[1]) - target_aspect))
    return best_bucket

def setup_models():
    print(f"Loading VAE: {VAE_MODEL_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(DEVICE).eval()
    print(f"Loading Text Encoder: {TEXT_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    full_model = AutoModel.from_pretrained(TEXT_MODEL_ID, trust_remote_code=True)
    text_model = full_model.encoder if hasattr(full_model, "encoder") else full_model
    text_model.to(DEVICE).eval()
    return vae, tokenizer, text_model

def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files_in_dir = os.listdir(OUTPUT_DIR)
    if len(files_in_dir) > 0:
        print(f"Overwriting cache in {OUTPUT_DIR} (cleaning old .pt files)...")
        for f in files_in_dir:
            if f.endswith('.pt'):
                os.remove(os.path.join(OUTPUT_DIR, f))
                
    vae, tokenizer, text_model = setup_models()
    
    print("Starting preprocessing...")
    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    bucket_counts = {b: 0 for b in BUCKETS}
    
    for filename in tqdm(files):
        try:
            img_path = os.path.join(DATASET_DIR, filename)
            txt_path = os.path.join(DATASET_DIR, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(txt_path):
                 txt_path = os.path.join(DATASET_DIR, os.path.splitext(filename)[0] + ".caption")
                 if not os.path.exists(txt_path): continue

            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            bw, bh = get_best_bucket(w, h)
            
            bucket_counts[(bw, bh)] += 1
            
            img = transforms.Resize((bh, bw), interpolation=transforms.InterpolationMode.LANCZOS)(image)
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
            img_tensor = TF.normalize(img_tensor, [0.5], [0.5])

            with torch.no_grad():
                latents = vae.encode(img_tensor).latent_dist.sample()
                latents = latents * VAE_SCALING_FACTOR 

            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            inputs = tokenizer(prompt, max_length=MAX_TOKEN_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                text_embeds = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

            save_data = {
                "latents": latents.squeeze(0).cpu().to(dtype=DTYPE),
                "text_embeds": text_embeds.squeeze(0).cpu().to(dtype=DTYPE),
                "width": bw,
                "height": bh
            }
            torch.save(save_data, os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".pt"))

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        
    print("\n" + "="*30)
    print(f"      STATS FOR {TARGET_RESOLUTION}px      ")
    print("="*30)
    total_images = sum(bucket_counts.values())
    for (bw, bh), count in bucket_counts.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        if count > 0:
            print(f"[{bw}x{bh}]: {count:3d} images ({percentage:.1f}%)")
    print("="*30)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    start_time = time.time()
    process()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")