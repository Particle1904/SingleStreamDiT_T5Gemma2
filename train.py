import os
import sys
import torch
import time
import re
import glob
import csv
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SingleStreamDiT
from diffusers import AutoencoderKL
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import get_cosine_schedule_with_warmup
from config import Config
from dataset import TextImageDataset, BucketBatchSampler
from latents import decode_latents_to_image
from samplers import run_sampling_pipeline
from losses import calculate_total_loss, prepare_batch_and_targets

# PATHS
CACHE_DIR = Config.cache_dir
CHECKPOINT_DIR = Config.checkpoint_dir
SAMPLES_DIR = Config.samples_dir
LOG_FILE = Config.log_file
TARGET_FILE = Config.target_file

# --- CONFIGURATION ---
PROJECT_NAME = Config.project_name
RESUME_FROM = Config.resume_from
BATCH_SIZE = Config.batch_size
LEARNING_RATE = Config.learning_rate
EPOCHS = Config.epochs
LOSS_TYPE = Config.loss_type
USE_SELF_EVAL = Config.use_self_eval
START_SELF_EVAL_AT = Config.start_self_eval_at
SELF_EVAL_LAMBDA = Config.self_eval_lambda
FAL_LAMBDA = Config.fal_lambda
FCL_LAMBDA = Config.fcl_lambda
FLIP_AUG = Config.flip_aug
SHIFT_VAL = Config.shift_val
SAVE_EVERY = Config.save_every
VALIDATE_EVERY = Config.validate_every
TEXT_DROPOUT = Config.text_dropout
EMA_DECAY = Config.ema_decay
ACCUM_STEPS = Config.accum_steps
GRADIENT_CHECKPOINTING = Config.gradient_checkpointing
IN_CHANNELS = Config.in_channels
WEIGHT_DECAY = Config.weight_decay
RESET_OPTIMIZER = Config.reset_optmizer
OPTIMIZER_WARMUP = Config.optimizer_warmup
OFFSET_NOISE = Config.offset_noise

# MODEL SETTINGS
DEVICE = Config.device
DTYPE = Config.dtype
VAE_ID = Config.vae_id
VAE_DOWNSAMPLE_FACTOR = Config.vae_downsample_factor

# VALIDATION SETTINGS
VALIDATE_CFG = Config.validate_cfg
VALIDATE_STEPS = Config.validate_steps 
VALIDATE_SAMPLER = Config.validate_sampler

# PERFORMANCE SETTINGS
LOAD_ENTIRE_DATASET = Config.load_entire_dataset
NUM_WORKERS = Config.num_workers

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

gate_learning_rate_factor = 10.0

def setup_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def cleanup_checkpoints(directory, prefix, keep_last_n=1):
    search_pattern = os.path.join(directory, f"{prefix}*.pt")
    files = glob.glob(search_pattern)
    
    if len(files) <= keep_last_n:
        return

    def get_epoch_num(filepath):
        match = re.search(r"epoch_(\d+)", filepath)
        if match:
            return int(match.group(1))
        return -1

    files.sort(key=get_epoch_num)

    files_to_delete = files[:-keep_last_n]

    for f in files_to_delete:
        try:
            os.remove(f)
            print(f"Cleanup: Deleted old checkpoint {os.path.basename(f)}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")

def get_gate_stats(model):
    if hasattr(model, "module"):
        model = model.module

    gate_values = []
    for _, module in model.named_modules():
        if hasattr(module, "gate") and isinstance(module.gate, torch.nn.Parameter):
            gate_values.append(module.gate.item())

    if len(gate_values) == 0:
        return 0.0, 0.0, 0.0

    avg_val = sum(gate_values) / len(gate_values)
    min_val = min(gate_values)
    max_val = max(gate_values)

    return avg_val, min_val, max_val

class CSVLogger:
    def __init__(self, filepath, resume=False):
        self.filepath = filepath
        self.resume = resume
        if not os.path.exists(filepath) or not resume:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Global_Step", "Loss", "LR", "Gate_Avg", "Gate_Min", "Gate_Max"])

    def log(self, epoch, step, loss, lr, gate_avg, gate_min, gate_max):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, lr, gate_avg, gate_min, gate_max])

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_size_mb = (total_params * 2) / (1024 * 1024)
    
    print(f"\n" + "="*50)
    print(f"      MODEL ARCHITECTURE STATISTICS      ")
    print(f"="*50)
    print(f"Total Parameters:      {total_params / 1e6:.2f}M")
    print(f"Trainable Parameters:  {trainable_params / 1e6:.2f}M")
    print(f"Weight File Size:      {param_size_mb:.2f} MB (at BF16)")
    print(f"Hidden Size:           {Config.hidden_size}")
    print(f"Depth:                 {Config.depth}")
    print(f"Attention Heads:       {Config.num_heads}")
    print(f"Patch Size:            {Config.patch_size}")
    print(f"="*50 + "\n")

@torch.no_grad()
def validate(model, vae, epoch, is_ema=False):
    model.eval()
    
    if not os.path.exists(TARGET_FILE):
        print(f"Validation Error: {TARGET_FILE} not found!")
        return

    data = torch.load(TARGET_FILE, map_location="cpu")
    h, w = data["height"], data["width"]    
    text_embeds = data["text_embeds"].unsqueeze(0).to(DEVICE, DTYPE)

    uncond_embeds = torch.zeros_like(text_embeds)
    combined_text_embeds = torch.cat([uncond_embeds, text_embeds], dim=0)

    torch_generator = torch.Generator(device=DEVICE).manual_seed(42)
    initial_noise = torch.randn(1, 16, h // VAE_DOWNSAMPLE_FACTOR, w // VAE_DOWNSAMPLE_FACTOR, 
                                generator=torch_generator, device=DEVICE, dtype=DTYPE)
    
    print(f"Validating {'EMA' if is_ema else 'RAW'}...")
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        final_latents = run_sampling_pipeline(model=model, initial_noise=initial_noise, steps=VALIDATE_STEPS, 
                                              combined_text_embeds=combined_text_embeds, cfg=VALIDATE_CFG, 
                                              sampler_type=VALIDATE_SAMPLER, shift_val=SHIFT_VAL)
            
    image = decode_latents_to_image(vae_model=vae, latents=final_latents, device=DEVICE)
    save_path = f"{SAMPLES_DIR}/{'EMA_' if is_ema else 'RAW_'}epoch_{epoch}.png"
    Image.fromarray(image[0]).save(save_path)
    model.train()

def train():
    setup_dirs()
    print(f"Loading DiT...")
    model = SingleStreamDiT(in_channels=IN_CHANNELS, gradient_checkpointing=GRADIENT_CHECKPOINTING).to(DEVICE)    
    
    print("Checking if Linux for torch.compile...")
    if sys.platform.startswith('linux'):
        print("Compiling model (Linux detected)...")
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Compilation successful.")
        except Exception as e:
            print(f"Compilation failed: {e}")
            print("Continuing without compilation.")
        
    print(f"Loading VAE: {VAE_ID}")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE)
    
    print_model_size(model)
    model.initialize_weights()
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    
    start_epoch = 0
    global_step = 0
    checkpoint_data = None
    
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Pre-loading checkpoint to determine schedule: {RESUME_FROM}")
        checkpoint_data = torch.load(RESUME_FROM, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.load_state_dict(checkpoint_data)
            
        if 'ema_state_dict' in checkpoint_data:
            ema_model.module.load_state_dict(checkpoint_data['ema_state_dict'])
        else:
            ema_model.update_parameters(model)

        start_epoch = checkpoint_data.get('epoch', 0) + 1
        global_step = checkpoint_data.get('global_step', 0)
    else:
        print("Starting from scratch (Epoch 0).")
    
    print("Initializing Dataset...")
    full_dataset = TextImageDataset()
    batch_sampler = BucketBatchSampler(buckets=full_dataset.buckets, batch_size=BATCH_SIZE, drop_last=True)
    dataloader = DataLoader(full_dataset, batch_sampler=batch_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    
    steps_per_epoch = len(dataloader) // ACCUM_STEPS
    remaining_epochs = EPOCHS - start_epoch    
    if remaining_epochs <= 0:
        total_steps = steps_per_epoch * 100 
    else:
        total_steps = steps_per_epoch * remaining_epochs
        
    warmup_steps = int(total_steps * OPTIMIZER_WARMUP)
    print(f"Schedule: {remaining_epochs} epochs left. Total steps: {total_steps}. Warmup: {warmup_steps}")

    #optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    param_base = []
    param_fourier_gates = []
    
    for name, param in model.named_parameters():
        if 'fourier_filter.gate' in name:
            param_fourier_gates.append(param)
        else:
            param_base.append(param)
            
    GATE_LEARNING_RATE = LEARNING_RATE * gate_learning_rate_factor     
    optimizer_grouped_parameters = [
        {'params': param_base, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
        {'params': param_fourier_gates, 'lr': GATE_LEARNING_RATE, 'weight_decay': 0.0},
    ]
    optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=LEARNING_RATE) 
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    if checkpoint_data is not None:
        if not RESET_OPTIMIZER and 'optimizer_state_dict' in checkpoint_data:
            print("Restoring Optimizer and Scheduler state...")
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        else:
            status = "Intentional Reset" if RESET_OPTIMIZER else "Fresh Start"
            print(f"{status}: Starting with fresh Optimizer/Scheduler curves.")
    
    del checkpoint_data 
    
    logger = CSVLogger(LOG_FILE, resume=(RESUME_FROM is not None))
    print(f"Training Start: {EPOCHS} Epochs, {total_steps} Steps, Shift={SHIFT_VAL}")

    for epoch in range(start_epoch, EPOCHS):
        pbar = tqdm(dataloader)
        optimizer.zero_grad()   
            
        for step, batch in enumerate(pbar):
            x_t, t, x_1, target, text = prepare_batch_and_targets(batch, DEVICE, DTYPE, SHIFT_VAL, OFFSET_NOISE)
            
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                if USE_SELF_EVAL and epoch > (EPOCHS * START_SELF_EVAL_AT):
                    # If Self-Eval is ON, the EMA forward pass must be wrapped in torch.no_grad()                    
                    with torch.no_grad():
                        loss = calculate_total_loss(
                            model, ema_model, x_t, t, x_1, target, text, epoch, EPOCHS, USE_SELF_EVAL,
                            START_SELF_EVAL_AT, SELF_EVAL_LAMBDA, FAL_LAMBDA, FCL_LAMBDA, LOSS_TYPE, 
                            ACCUM_STEPS)
                else:
                    loss = calculate_total_loss(
                        model, ema_model, x_t, t, x_1, target, text, epoch, EPOCHS, USE_SELF_EVAL,
                        START_SELF_EVAL_AT, SELF_EVAL_LAMBDA, FAL_LAMBDA, FCL_LAMBDA, LOSS_TYPE, 
                        ACCUM_STEPS)

            loss.backward()
            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ema_model.update_parameters(model)
                global_step += 1
            
            current_loss = loss.item() * ACCUM_STEPS
            if step % 10 == 0:
                lr_current = optimizer.param_groups[0]['lr']
                avg_gate, min_gate, max_gate = get_gate_stats(model)      
                self_eval_status = "On" if USE_SELF_EVAL and epoch > (EPOCHS * START_SELF_EVAL_AT) else "Off"
                pbar.set_description(f"Ep {epoch}|Loss: {current_loss:.3f}|LR: {lr_current:.6f}|Gate(avg-min-max): {avg_gate:.3f}[{min_gate:.3f}/{max_gate:.3f}]|Self-E: {self_eval_status}|")
                logger.log(epoch, global_step, current_loss, lr_current, avg_gate, min_gate, max_gate)

        if epoch > 0 and epoch % VALIDATE_EVERY == 0:
            validate(ema_model.module, vae, epoch, is_ema=True)
            validate(model, vae, epoch, is_ema=False)
            
        if epoch > 0 and epoch % SAVE_EVERY == 0:
            save_data = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            # Save new and delete old.
            torch.save(save_data, f"{CHECKPOINT_DIR}/full_state_epoch_{epoch}.pt")
            cleanup_checkpoints(CHECKPOINT_DIR, "full_state_", keep_last_n=1)
            
            # Save new and delete old, keeping last 3 EMA
            torch.save(ema_model.module.state_dict(), f"{CHECKPOINT_DIR}/ema_weights_epoch_{epoch}.pt")
            cleanup_checkpoints(CHECKPOINT_DIR, "ema_weights_", keep_last_n=3)
            
    print(f"Loop Finished. Forcing save at Epoch {EPOCHS}...")
    
    validate(ema_model.module, vae, EPOCHS, is_ema=True)
    save_data = {
        'epoch': EPOCHS,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(save_data, f"{CHECKPOINT_DIR}/full_state_final.pt")
    torch.save(ema_model.module.state_dict(), f"{CHECKPOINT_DIR}/ema_weights_final.pt")
    
if __name__ == "__main__":
    start_time = time.time()
    train()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")
    