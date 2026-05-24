# scripts/check_pt.py
import torch
import sys
import os
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config
from latents import load_repa_target

def inspect():
    print(f"Checking {Config.target_file}...")
    
    data = torch.load(Config.target_file, map_location="cpu")
    
    print("\n" + "="*45)
    print("           PT FILE DETAILED INFO")
    print("="*45)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            num_bytes = v.element_size() * v.numel()
            size_kb = num_bytes / 1024
            print(f"Key: {k:<20} | Shape: {str(list(v.shape)):<22} | Dtype: {str(v.dtype):<15} | Disk Size: {size_kb:.2f} KB")
        else:
            print(f"Key: {k:<20} | Value: {v}")
    print("="*45 + "\n")
    
    repa_target_raw = data.get("repa_target", None)
    if repa_target_raw is not None:
        print("="*45)
        print("        QUANTIZATION INTEGRITY CHECK")
        print("="*45)
        print(f"Raw Int8 Tensor shape: {repa_target_raw.shape}")
        print(f"Raw Int8 Min Value  : {repa_target_raw.min().item()}")
        print(f"Raw Int8 Max Value  : {repa_target_raw.max().item()}")
        
        repa_target_float = load_repa_target(data, torch.float32)
        print("\n--- Dequantized Float32 ---")
        print(f"Float32 Target shape : {repa_target_float.shape}")
        print(f"Float32 Min Value    : {repa_target_float.min().item():.6f}")
        print(f"Float32 Max Value    : {repa_target_float.max().item():.6f}")
        print(f"Float32 Mean Value   : {repa_target_float.mean().item():.6f}")
        print("="*45 + "\n")
        
        print("Calculating Semantic PCA Maps from DINOv3 features...")
        width, height = data["width"], data["height"]
        grid_w = width // 16
        grid_h = height // 16
        
        visualize_pca(repa_target_raw.to(torch.float32), grid_h, grid_w, "dino_features_int8_pca.png")
        print("Saved raw INT8 PCA map to: dino_features_int8_pca.png")
        
        visualize_pca(repa_target_float, grid_h, grid_w, "dino_features_float32_pca.png")
        print("Saved dequantized FLOAT32 PCA map to: dino_features_float32_pca.png")
        print("\nCompare both images! They will look identical because symmetric scaling")
        print("preserves the semantic eigenvector space perfectly.")

def visualize_pca(features, grid_h, grid_w, filename):
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    
    U, S, V = torch.linalg.svd(centered, full_matrices=False)
    
    pca_proj = torch.matmul(centered, V[:3, :].t())
    
    min_val = pca_proj.min(dim=0, keepdim=True).values
    max_val = pca_proj.max(dim=0, keepdim=True).values
    normalized = (pca_proj - min_val) / (max_val - min_val + 1e-5)
    
    rgb_map = normalized.view(grid_h, grid_w, 3).numpy()
    rgb_map = (rgb_map * 255.0).astype("uint8")
    
    img = Image.fromarray(rgb_map)
    img_large = img.resize((grid_w * 10, grid_h * 10), resample=Image.NEAREST)
    img_large.save(filename)

if __name__ == "__main__":
    inspect()