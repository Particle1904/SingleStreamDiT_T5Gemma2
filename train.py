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
LOG_EVERY_STEPS = 10

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

@torch.no_grad()
def calculate_validation_loss(accelerator, model, val_loader, epoch):
    model.eval()
    total_validate_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        x_t, t, x_1, target, text = prepare_batch_and_targets(batch, Config.device, Config.dtype, Config.shift_val, Config.offset_noise)
        
        loss = calculate_total_loss(
            model, None, x_t, t, x_1, target, text, epoch, Config.epochs, use_self_eval=False, 
            start_self_eval_at=1.0, self_eval_lambda=0, fal_lambda=Config.fal_lambda, 
            fcl_lambda=Config.fcl_lambda, loss_type=Config.loss_type, accum_steps=1)
        total_validate_loss += loss.item()
        num_batches += 1
    
    if num_batches > 0:
        avg_val_loss = total_validate_loss / num_batches
    else:
        avg_val_loss = 0.0
    avg_val_loss = accelerator.gather(torch.tensor(avg_val_loss, device=Config.device).unsqueeze(0)).mean().item()
    
    model.train()
    return avg_val_loss

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
def validate(accelerator, model, vae, epoch, global_step, is_ema=False):
    if not accelerator.is_main_process:
        return
    model.eval()
    
    if not os.path.exists(Config.target_file):
        print(f"Validation Error: {Config.target_file} not found!")
        return

    data = torch.load(Config.target_file, map_location="cpu")
    h, w = data["height"], data["width"]    
    text_embeds = data["text_embeds"].unsqueeze(0).to(Config.device, Config.dtype)

    uncond_embeds = torch.zeros_like(text_embeds)
    combined_text_embeds = torch.cat([uncond_embeds, text_embeds], dim=0)

    torch_generator = torch.Generator(device=Config.device).manual_seed(Config.seed)
    initial_noise = torch.randn(1, 16, h // Config.vae_downsample_factor, w // Config.vae_downsample_factor, 
                                generator=torch_generator, device=Config.device, dtype=Config.dtype)
    
    print(f"Validating {'EMA' if is_ema else 'RAW'}...")
    with torch.autocast(device_type="cuda", dtype=Config.dtype):
        final_latents = run_sampling_pipeline(model=model, initial_noise=initial_noise,
                                              steps=Config.validate_steps, 
                                              combined_text_embeds=combined_text_embeds, 
                                              cfg=Config.validate_cfg, sampler_type=Config.validate_sampler, 
                                              shift_val=Config.shift_val)
       
    if accelerator.is_main_process:
        image = decode_latents_to_image(vae_model=vae, latents=final_latents, device=Config.device)
        accelerator.get_tracker("wandb").log(
            {"validation_sample": wandb.Image(image)}, step=global_step
        )
        save_path = f"{Config.samples_dir}/{'EMA_' if is_ema else 'RAW_'}epoch_{epoch}.png"
        image.save(save_path)
    model.train()

def train():
    # ---- accelerator & logging ----
    accelerator = Config.accelerator
    if accelerator.is_main_process:
        setup_dirs()
        accelerator.init_trackers(project_name=Config.project_name,
            config={k: v for k, v in Config.__dict__.items() if not k.startswith("__")})
        
    # ---- model & VAE ----
    print(f"Loading DiT...")
    model = SingleStreamDiT(in_channels=Config.in_channels, 
                            gradient_checkpointing=Config.gradient_checkpointing).to(Config.device)    
         
    print(f"Loading VAE: {Config.vae_id}")
    vae = AutoencoderKL.from_pretrained(Config.vae_id).to(Config.device)
    
    print_model_size(model)
    model.initialize_weights()
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(Config.ema_decay))
    
    # ---- checkpoint restore ----
    start_epoch = 0
    global_step = 0
    checkpoint_data = None
    
    if Config.resume_from and os.path.exists(Config.resume_from):
        print(f"Pre-loading checkpoint to determine schedule: {Config.resume_from}")
        checkpoint_data = torch.load(Config.resume_from, map_location=Config.device)
        
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            model.to(Config.dtype)
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
    
    # ---- dataset & dataloaders ----
    print("Initializing Dataset...")
    full_dataset = TextImageDataset()
    
    if Config.run_validation_loss:
        print("Validation Enabled: Splitting dataset (holding out data)...")
        train_idx_set, val_idx_set = split_dataset_indices(len(full_dataset), items_per_category=20, 
                                                           val_per_category=4)
        print(f"Dataset Split: {len(train_idx_set)} Train, {len(val_idx_set)} Validation")
    else:
        print("Validation Disabled: Using FULL dataset for training (0 holdout).")
        train_idx_set = set(range(len(full_dataset)))
        val_idx_set = set() 
           
    print(f"Dataset Split: {len(train_idx_set)} Train, {len(val_idx_set)} Validation")
    
    train_buckets = {}
    val_buckets = {}
    
    for res, indices in full_dataset.buckets.items():
        t_list = [i for i in indices if i in train_idx_set]
        v_list = [i for i in indices if i in val_idx_set]
        if t_list: train_buckets[res] = t_list
        if v_list: val_buckets[res] = v_list
        
    train_sampler = BucketBatchSampler(train_buckets, batch_size=Config.batch_size, drop_last=True)
    if len(val_buckets) > 0:
        val_sampler = BucketBatchSampler(val_buckets, batch_size=Config.batch_size, drop_last=False)
        val_loader = DataLoader(full_dataset, batch_sampler=val_sampler, num_workers=Config.num_workers, 
                                pin_memory=True)
    else:
        val_loader = None
    
    train_loader = DataLoader(full_dataset, batch_sampler=train_sampler, num_workers=Config.num_workers, 
                              pin_memory=True)
    dataloader = train_loader
    
    steps_per_epoch = len(dataloader) // Config.accum_steps
    remaining_epochs = Config.epochs - start_epoch    
    if remaining_epochs <= 0:
        total_steps = steps_per_epoch * 100 
    else:
        total_steps = steps_per_epoch * remaining_epochs
        
    warmup_steps = int(total_steps * Config.optimizer_warmup)
    print(f"Schedule: {remaining_epochs} epochs left. Total steps: {total_steps}. Warmup: {warmup_steps}")

    # ---- optimizer & scheduler ----
    param_base = []
    param_fourier_gates = []
    
    for name, param in model.named_parameters():
        if 'fourier_filter.gate' in name:
            param_fourier_gates.append(param)
        else:
            param_base.append(param)
            
    gate_learning_rate = Config.learning_rate * GATE_LEARNING_RATE_MULTIPLIER     
    optimizer_grouped_parameters = [
        {'params': param_base, 'lr': Config.learning_rate, 'weight_decay': Config.weight_decay},
        {'params': param_fourier_gates, 'lr': gate_learning_rate, 'weight_decay': 0.0},
    ]
    optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=Config.learning_rate) 
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
       
    print("Checking if Linux for torch.compile...")
    if sys.platform.startswith('linux'):
        print("Compiling model (Linux detected)...")
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Compilation successful.")
        except Exception as e:
            print(f"Compilation failed: {e}")
            print("Continuing without compilation.")

    if checkpoint_data is not None:
        if not Config.reset_optimizer and 'optimizer_state_dict' in checkpoint_data:
            print("Restoring Optimizer and Scheduler state...")
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        else:
            status = "Intentional Reset" if Config.reset_optimizer else "Fresh Start"
            print(f"{status}: Starting with fresh Optimizer/Scheduler curves.")
    
    del checkpoint_data 
    
    logger = CSVLogger(Config.log_file, resume=(Config.resume_from is not None))
    print(f"Training Start: {Config.epochs} epochs, {total_steps} Steps, Shift={Config.shift_val}")

    # ---- training loop ----
    for epoch in range(start_epoch, Config.epochs):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process, mininterval=30)
        optimizer.zero_grad()   
            
        for step, batch in enumerate(pbar):
            x_t, t, x_1, target, text = prepare_batch_and_targets(batch, Config.device, Config.dtype, Config.shift_val, Config.offset_noise)
            
            self_eval_active = (Config.use_self_eval and epoch > (Config.epochs * Config.start_self_eval_at))
            
            with accelerator.autocast():
                if self_eval_active:
                        loss = calculate_total_loss(
                            model, ema_model, x_t, t, x_1, target, text, epoch, Config.epochs, Config.use_self_eval,
                            Config.start_self_eval_at, Config.self_eval_lambda, Config.fal_lambda, Config.fcl_lambda, Config.loss_type, 
                            Config.accum_steps)
                else:
                    loss = calculate_total_loss(
                        model, ema_model, x_t, t, x_1, target, text, epoch, Config.epochs, Config.use_self_eval,
                        Config.start_self_eval_at, Config.self_eval_lambda, Config.fal_lambda, Config.fcl_lambda, Config.loss_type, 
                        Config.accum_steps)

            accelerator.backward(loss)
            if (step + 1) % Config.accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    ema_model.update_parameters(accelerator.unwrap_model(model))
                global_step += 1
            
            if accelerator.is_main_process:
                current_loss = loss.item() * Config.accum_steps
                if step % LOG_EVERY_STEPS == 0:
                    lr_current = optimizer.param_groups[0]['lr']
                    avg_gate, min_gate, max_gate = get_gate_stats(model)      
                    self_eval_status = "On" if self_eval_active else "Off"
                    pbar.set_description(f"Ep {epoch}|Loss: {current_loss:.3f}|LR: {lr_current:.6f}|Gate(avg-min-max): {avg_gate:.3f}[{min_gate:.3f}/{max_gate:.3f}]|Self-E: {self_eval_status}|")
                    logger.log(epoch, global_step, current_loss, lr_current, avg_gate, min_gate, max_gate)
                    accelerator.log({"train_loss": current_loss,
                                     "learning_rate": lr_current, 
                                     "gate_avg": avg_gate,
                                     "gate_min": min_gate,
                                     "gate_max": max_gate},
                                    step=global_step)

        if epoch > 0 and epoch % Config.validate_every == 0:
            validate(accelerator, model, vae, epoch, global_step, is_ema=False)
            validate(accelerator, ema_model.module, vae, epoch, global_step, is_ema=True)
            if Config.run_validation_loss and val_loader is not None:
                val_loss = calculate_validation_loss(accelerator, accelerator.unwrap_model(model), val_loader, epoch)
                if accelerator.is_main_process:
                    print(f"Epoch {epoch} | Validation Loss: {val_loss:.5f}")
                    accelerator.log({"val_loss": val_loss}, step=global_step)
            
        if accelerator.is_main_process and epoch > 0 and epoch % Config.save_every == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            save_data = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': unwrapped_model.state_dict(),
                'ema_state_dict': ema_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(save_data, f"{Config.checkpoint_dir}/full_state_epoch_{epoch}.pt")
            cleanup_checkpoints(Config.checkpoint_dir, "full_state_", keep_last_n=1)
            
            torch.save(ema_model.module.state_dict(), f"{Config.checkpoint_dir}/ema_weights_epoch_{epoch}.pt")
            cleanup_checkpoints(Config.checkpoint_dir, "ema_weights_", keep_last_n=3)
            
            
    if accelerator.is_main_process:
        print(f"Loop Finished. Forcing save at Epoch {Config.epochs}...")
        validate(accelerator, ema_model.module, vae, Config.epochs, global_step, is_ema=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        save_data = {
            'epoch': Config.epochs,
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'ema_state_dict': ema_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(save_data, f"{Config.checkpoint_dir}/full_state_final.pt")
        torch.save(ema_model.module.state_dict(), f"{Config.checkpoint_dir}/ema_weights_final.pt")
        accelerator.end_training()
           
if __name__ == "__main__":
    start_time = time.time()
    train()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")
    