from PIL import Image
import os
import re
import sys
# Import config from the folder above.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config

def generate_samples_gif(output_filename="training_progression.webp", duration=100):
    samples_dir = Config.samples_dir
    
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found at {samples_dir}")
        return

    pattern = re.compile(r"RAW_epoch_(\d+)\.png")
    
    sample_files = []
    for f in os.listdir(samples_dir):
        match = pattern.search(f)
        if match:
            epoch_num = int(match.group(1))
            sample_files.append((epoch_num, os.path.join(samples_dir, f)))
    
    sample_files.sort(key=lambda x: x[0])
    
    if not sample_files:
        print("No RAW sample images found to create GIF.")
        return

    print(f"Found {len(sample_files)} samples. Creating GIF...")
    
    frames = []
    for _, path in sample_files:
        img = Image.open(path).convert("RGB")
        frames.append(img)
    
    last_frame = frames[-1]
    for _ in range(10):
        frames.append(last_frame)
    
    frames[0].save(output_filename, format="WEBP", append_images=frames[1:], save_all=True, duration=duration, loop=0, lossless=False, quality=90)
    
    print(f"Successfully saved animation to: {output_filename}")

if __name__ == "__main__":
    generate_samples_gif()