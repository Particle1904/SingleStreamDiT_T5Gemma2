import os
import sys
import torch
import glob
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import AutoModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)

from config import Config

def build_image_map(dataset_dir):
    image_map = {}
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(valid_exts):
                base = os.path.splitext(file)[0]
                image_map[base] = os.path.join(root, file)
    return image_map

def process_cache():
    print(f"Scanning cache directory: {Config.cache_dir}")
    pt_files = glob.glob(os.path.join(Config.cache_dir, "*.pt"))
    
    if not pt_files:
        print("No cache files found.")
        return
        
    print(f"Indexing raw images recursively from: {Config.dataset_dir}")
    image_map = build_image_map(Config.dataset_dir)
    print(f"Successfully indexed {len(image_map)} raw images.")
    
    bucket_groups = {}
    files_to_process = 0
    
    print("Checking which files need REPA targets...")
    for pt_file in tqdm(pt_files, desc="Scanning metadata"):
        data = torch.load(pt_file, map_location="cpu")
        if "repa_target" not in data:
            res_key = (data["width"], data["height"])
            if res_key not in bucket_groups:
                bucket_groups[res_key] = []
            bucket_groups[res_key].append(pt_file)
            files_to_process += 1
        del data

    if files_to_process == 0:
        print("All cache files already have 'repa_target'. Nothing to do!")
        return

    repa_model_name = getattr(Config, "repa_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    print(f"Loading {repa_model_name}...")
    dinov3 = AutoModel.from_pretrained(repa_model_name).to(Config.device, dtype=Config.dtype).eval()
    
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=Config.device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=Config.device).view(1, 3, 1, 1)

    batch_size = 64
    progress_bar = tqdm(total=files_to_process, desc="Updating Cache")
    
    skipped_count = 0
    
    for res_key, paths in bucket_groups.items():
        bw, bh = res_key
        
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_tensors = []
            valid_paths = []
            
            for pt_path in batch_paths:
                basename = os.path.splitext(os.path.basename(pt_path))[0]
                img_path = image_map.get(basename)
                
                if not img_path: 
                    skipped_count += 1
                    continue
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_w, img_h = img.size
                    img_aspect = img_w / img_h
                    target_aspect = bw / bh
                    
                    resize_w, resize_h = (int(bh * img_aspect), bh) if img_aspect > target_aspect else (bw, int(bw / img_aspect))
                    img = img.resize((resize_w, resize_h), resample=Image.LANCZOS)
                    
                    left, top = (resize_w - bw) // 2, (resize_h - bh) // 2
                    img = img.crop((left, top, left + bw, top + bh))
                    batch_tensors.append(TF.to_tensor(img))
                    valid_paths.append(pt_path)
                except Exception as e:
                    print(f"Error on {img_path}: {e}")
                    
            if not batch_tensors:
                progress_bar.update(len(batch_paths))
                continue
                
            img_tensors_01 = torch.stack(batch_tensors).to(Config.device)
            dinov3_input = (img_tensors_01 - imagenet_mean) / imagenet_std
            
            device_type = Config.device.type if isinstance(Config.device, torch.device) else torch.device(Config.device).type
            
            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=Config.dtype):
                dino_out = dinov3(dinov3_input)
                
                num_patches = (bh // 16) * (bw // 16)
                repa_targets = dino_out.last_hidden_state[:, -num_patches:, :]

            for j, pt_path in enumerate(valid_paths):
                data = torch.load(pt_path, map_location="cpu")
                
                tensor_to_save = repa_targets[j].cpu()
                scale = tensor_to_save.abs().max()
                if scale == 0:
                    scale = torch.tensor(1.0)
                scale_factor = scale / 127.0
                quantized_tensor = (tensor_to_save / scale_factor).round().clamp(-128, 127).to(torch.int8)
                
                data["repa_target"] = quantized_tensor
                data["repa_scale"] = scale_factor.to(torch.float32)
                
                tmp_path = pt_path + ".tmp"
                torch.save(data, tmp_path)
                os.replace(tmp_path, pt_path)
                
            progress_bar.update(len(batch_paths))
            
    progress_bar.close()
    
    if skipped_count > 0:
        print(f"\n[WARNING] Skipped {skipped_count} cache files because their corresponding raw images were not found inside {Config.dataset_dir}!")
    else:
        print("\nAll cache files processed successfully!")

if __name__ == "__main__":
    process_cache()