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
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
from config import Config
from dataset import TextImageDataset, BucketBatchSampler, split_dataset_indices
from latents import decode_latents_to_image
from samplers import run_sampling_pipeline
from losses import calculate_total_loss, prepare_batch_and_targets
import wandb
import builtins

if not Config.accelerator.is_main_process:
    def print_pass(*args, **kwargs): pass
    builtins.print = print_pass

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

GATE_LEARNING_RATE_MULTIPLIER = 1

LOG_EVERY_STEPS = 1 

def setup_dirs():
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.samples_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.log_file), exist_ok=True)

def cleanup_checkpoints(directory, prefix, keep_last_n=1):
    search_pattern = os.path.join(directory, f"{prefix}*.pt")
    files = glob.glob(search_pattern)
    if len(files) <= keep_last_n: 
        return
    
    def get_epoch_num(filepath):
        match = re.search(r"epoch_(\d+)", filepath)
        return int(match.group(1)) if match else -1
    files.sort(key=get_epoch_num)
    for f in files[:-keep_last_n]:
        try: os.remove(f)
        except: 
            pass

def get_gate_stats(model):
    m = model.module if hasattr(model, "module") else model
    gate_values = [module.gate.item() for _, module in m.named_modules() 
                   if hasattr(module, "gate") and isinstance(module.gate, torch.nn.Parameter)]
    if not gate_values: 
        return 0.0, 0.0, 0.0
    return sum(gate_values)/len(gate_values), min(gate_values), max(gate_values)

@torch.no_grad()
def calculate_validation_loss(accelerator, model, val_loader, epoch):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        x_t, t, x_1, target, text = prepare_batch_and_targets(batch, Config.device, Config.dtype, Config.shift_val, Config.offset_noise)
        loss = calculate_total_loss(model, None, x_t, t, x_1, target, text, epoch, Config.epochs, loss_type=Config.loss_type)
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / max(num_batches, 1)
    avg_loss = accelerator.gather(torch.tensor(avg_loss, device=Config.device).unsqueeze(0)).mean().item()
    model.train()
    return avg_loss

class CSVLogger:
    def __init__(self, filepath, resume=False):
        self.filepath = filepath
        if not os.path.exists(filepath) or not resume:
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(["Epoch", "Global_Step", "Loss", "LR", "Gate_Avg", "Gate_Min", "Gate_Max"])
    def log(self, epoch, step, loss, lr, gate_avg, gate_min, gate_max):
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, loss, lr, gate_avg, gate_min, gate_max])

@torch.no_grad()
def validate(accelerator, model, vae, epoch, global_step, is_ema=False):
    if not accelerator.is_main_process: 
        return
    model.eval()
    if not os.path.exists(Config.target_file): 
        return

    data = torch.load(Config.target_file, map_location="cpu")
    h, w = data["height"], data["width"]    
    text_embeds = data["text_embeds"].unsqueeze(0).to(Config.device, Config.dtype)
    uncond_embeds = torch.zeros_like(text_embeds)
    combined_text_embeds = torch.cat([uncond_embeds, text_embeds], dim=0)

    torch_generator = torch.Generator(device=Config.device).manual_seed(Config.seed)
    initial_noise = torch.randn(1, Config.in_channels, h // Config.vae_downsample_factor, w // Config.vae_downsample_factor, 
                                generator=torch_generator, device=Config.device, dtype=Config.dtype)
    
    print(f"Validating {'EMA' if is_ema else 'RAW'}...")
    with torch.autocast(device_type="cuda", dtype=Config.dtype):
        final_latents = run_sampling_pipeline(model=model, initial_noise=initial_noise, steps=Config.validate_steps, 
                                              combined_text_embeds=combined_text_embeds, cfg=Config.validate_cfg, 
                                              sampler_type=Config.validate_sampler, shift_val=Config.shift_val)
       
    image = decode_latents_to_image(vae_model=vae, latents=final_latents, device=Config.device)
    
    accelerator.get_tracker("wandb").log({"validation_sample": wandb.Image(image)}, step=global_step)
    image.save(f"{Config.samples_dir}/{'EMA_' if is_ema else 'RAW_'}epoch_{epoch}.png")
    model.train()

def train():
    accelerator = Config.accelerator
    
    if accelerator.is_main_process:
        setup_dirs()
    
    accelerator.init_trackers(
        project_name=Config.project_name,
        config={k: v for k, v in Config.__dict__.items() if not k.startswith("__")}
    )
        
    print(f"Loading DiT & VAE...")
    model = SingleStreamDiT(in_channels=Config.in_channels, gradient_checkpointing=Config.gradient_checkpointing).to(Config.device)    
    
    vae = None  
    if "FLUX.2" in Config.vae_id:
        print("Loading FLUX2 VAE...")
        vae = DiffusersAutoencoderKL.from_pretrained(Config.vae_id).to(Config.device).eval()
    else:
        print("Loading generic VAE...")
        vae = AutoencoderKL.from_pretrained(Config.vae_id).to(Config.device).eval()
    
    model.initialize_weights()
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(Config.ema_decay))
    
    start_epoch = 0
    global_step = 0
    checkpoint_data = None
    
    if Config.resume_from and os.path.exists(Config.resume_from):
        checkpoint_data = torch.load(Config.resume_from, map_location=Config.device)
        model.load_state_dict(checkpoint_data.get('model_state_dict', checkpoint_data))
        if 'ema_state_dict' in checkpoint_data:
            ema_model.module.load_state_dict(checkpoint_data['ema_state_dict'])
        start_epoch = checkpoint_data.get('epoch', 0) + 1
        global_step = checkpoint_data.get('global_step', 0)
    
    full_dataset = TextImageDataset()
    train_idx_set = set(range(len(full_dataset)))
    val_loader = None
    
    train_buckets = {res: [i for i in idxs if i in train_idx_set] for res, idxs in full_dataset.buckets.items() if any(i in train_idx_set for i in idxs)}
    train_loader = DataLoader(full_dataset, batch_sampler=BucketBatchSampler(train_buckets, batch_size=Config.batch_size), num_workers=Config.num_workers)
    
    steps_per_epoch = len(train_loader) // Config.accum_steps
    total_steps = steps_per_epoch * (Config.epochs - start_epoch)
    warmup_steps = int(total_steps * Config.optimizer_warmup)

    param_base = [p for n, p in model.named_parameters() if 'fourier_filter.gate' not in n]
    param_gates = [p for n, p in model.named_parameters() if 'fourier_filter.gate' in n]
    optimizer = bnb.optim.AdamW8bit([
        {'params': param_base, 'lr': Config.learning_rate, 'weight_decay': Config.weight_decay},
        {'params': param_gates, 'lr': Config.learning_rate * GATE_LEARNING_RATE_MULTIPLIER, 'weight_decay': 0.0},
    ], lr=Config.learning_rate) 
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
       
    if sys.platform.startswith('linux'):
        try:
            model = torch.compile(model, mode="max-autotune")
        except: 
            pass

    logger = CSVLogger(Config.log_file, resume=(Config.resume_from is not None))

    for epoch in range(start_epoch, Config.epochs):
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
        optimizer.zero_grad()   
            
        for step, batch in enumerate(pbar):
            x_t, t, x_1, target, text = prepare_batch_and_targets(batch, Config.device, Config.dtype, Config.shift_val, Config.offset_noise)
            
            with accelerator.autocast():
                loss = calculate_total_loss(model, ema_model, x_t, t, x_1, target, text, epoch, Config.epochs, 
                                            Config.use_self_eval, Config.start_self_eval_at, Config.self_eval_lambda, 
                                            Config.fal_lambda, Config.fcl_lambda, Config.loss_type, Config.accum_steps)

            accelerator.backward(loss)
            
            if (step + 1) % Config.accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if accelerator.is_main_process:
                    ema_model.update_parameters(accelerator.unwrap_model(model))
                
                if global_step % LOG_EVERY_STEPS == 0:
                    lr_curr = optimizer.param_groups[0]['lr']
                    avg_g, min_g, max_g = get_gate_stats(model)      
                    curr_loss = loss.item() * Config.accum_steps
                    
                    pbar.set_description(f"Ep {epoch}|Step {global_step}|Loss: {curr_loss:.3f}")
                    
                    if accelerator.is_main_process:
                        logger.log(epoch, global_step, curr_loss, lr_curr, avg_g, min_g, max_g)
                    
                    accelerator.log({
                        "train_loss": curr_loss,
                        "learning_rate": lr_curr, 
                        "gate_avg": avg_g,
                        "epoch": epoch
                    }, step=global_step)

        if epoch > 0 and epoch % Config.validate_every == 0:
            validate(accelerator, model, vae, epoch, global_step, is_ema=False)
            validate(accelerator, ema_model.module, vae, epoch, global_step, is_ema=True)
            
        if accelerator.is_main_process and epoch > 0 and epoch % Config.save_every == 0:
            unwrapped = accelerator.unwrap_model(model)
            save_path = f"{Config.checkpoint_dir}/full_state_epoch_{epoch}.pt"
            torch.save({'epoch': epoch, 'global_step': global_step, 'model_state_dict': unwrapped.state_dict(),
                        'ema_state_dict': ema_model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, save_path)
            cleanup_checkpoints(Config.checkpoint_dir, "full_state_", keep_last_n=1)

    accelerator.end_training()
           
if __name__ == "__main__":
    train()