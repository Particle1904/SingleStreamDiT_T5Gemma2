import torch
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config

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

if __name__ == "__main__":
    inspect()