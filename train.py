import os
import sys
import torch
import csv
import warnings
import bitsandbytes as bnb
import wandb
import builtins
import time
import copy
import math
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from accelerate.utils import find_executable_batch_size
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import get_cosine_schedule_with_warmup
from model import SingleStreamDiT
from config import Config, parse_config_args
from dataset import TextImageDataset, BucketBatchSampler
from latents import decode_latents_to_image
from model_loader import load_vae
from samplers import run_sampling_pipeline
from losses import calculate_total_loss, prepare_batch_and_targets
from checkpoint_manager import CheckpointManager
from utilities import parse_run_name, print_model_parameters

warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

LOG_EVERY_STEPS = 1 

parse_config_args()
Config.target_file = os.path.join(Config.cache_dir, Config.target_filename)
Config.checkpoint_dir = os.path.join(Config.output_dir, "checkpoints")
Config.samples_dir = os.path.join(Config.output_dir, "samples")
Config.log_file = os.path.join(Config.output_dir, "logs", f"{Config.project_name}_log.csv")

def can_attempt_resume(resume_value):
    return resume_value is not None

class CSVLogger:
    def __init__(self, filepath, resume=False):
        self.filepath = filepath
        if not os.path.exists(filepath) or not resume:
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(["Epoch", "Global_Step", "Loss", "LR"])
    def log(self, epoch, step, loss, lr):
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, loss, lr])

@torch.no_grad()
def validate(accelerator, model, vae, epoch, global_step, is_ema=False):
    if not accelerator.is_main_process: 
        return
    model.eval()
    if not os.path.exists(Config.target_file): 
        return

    data = torch.load(Config.target_file, map_location="cpu")
    h, w = data["height"], data["width"]    
    if "text_embeds_list" in data:
        text_embeds = data["text_embeds_list"][0].unsqueeze(0).to(Config.device, Config.dtype)
        text_mask = data["attention_mask_list"][0].unsqueeze(0).to(Config.device)
    else:
        text_embeds = data["text_embeds"].unsqueeze(0).to(Config.device, Config.dtype)
        text_mask = data["attention_mask"].unsqueeze(0).to(Config.device)
    uncond_embeds = torch.zeros_like(text_embeds)
    uncond_mask = torch.ones_like(text_mask)
    combined_text_embeds = torch.cat([uncond_embeds, text_embeds], dim=0)
    combined_mask = torch.cat([uncond_mask, text_mask], dim=0)

    torch_generator = torch.Generator(device=Config.device).manual_seed(Config.seed)
    initial_noise = torch.randn(1, Config.in_channels, h // Config.vae_downsample_factor, 
                                w // Config.vae_downsample_factor, 
                                generator=torch_generator, device=Config.device, dtype=Config.dtype)
    
    print(f"Validating {'EMA' if is_ema else 'RAW'}...")
    with torch.autocast(device_type="cuda", dtype=Config.dtype):
        final_latents = run_sampling_pipeline(model=model, initial_noise=initial_noise, 
                                              steps=Config.validate_steps, 
                                              combined_text_embeds=combined_text_embeds, cfg=Config.validate_cfg, 
                                              text_mask=combined_mask, sampler_type=Config.validate_sampler, 
                                              scheduler_type=Config.validate_scheduler, shift_val=Config.shift_val)
       
    image = decode_latents_to_image(vae_model=vae, latents=final_latents, device=Config.device)
    
    key_name = "validation_sample_ema" if is_ema else "validation_sample_raw"
    accelerator.get_tracker("wandb").log({key_name: wandb.Image(image)}, step=global_step)
    
    image.save(f"{Config.samples_dir}/{'EMA_' if is_ema else 'RAW_'}epoch_{epoch}.png")
    if not is_ema:
        model.train()

def train():
    checkpoint_manager = CheckpointManager(Config)
    mixed_precision_string = "fp16" if Config.dtype == torch.float16 else "bf16" if Config.dtype == torch.bfloat16 else "no"
    accelerator = Accelerator(log_with="wandb", mixed_precision=mixed_precision_string, 
                              gradient_accumulation_steps=Config.accum_steps,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    set_seed(Config.seed)
    Config.device = accelerator.device
    Config.accelerator = accelerator
    
    if not Config.accelerator.is_main_process:
        def print_pass(*args, **kwargs): pass
        builtins.print = print_pass
        
    if accelerator.is_main_process:
        checkpoint_manager.setup_dirs()
    
    wandb_run_id = None
    resume_flag = "allow"
    
    if can_attempt_resume(Config.resume_from):
        resolved_path = checkpoint_manager._resolve_path(Config.resume_from)
        if resolved_path is not None:
            found_id = checkpoint_manager.load_run_id(resolved_path)
            if found_id:
                wandb_run_id = found_id
                print(f"Found WandB Run ID in checkpoint: {wandb_run_id}")
    
    accelerator.init_trackers(project_name=Config.project_name, 
                              config={k: v for k, v in Config.__dict__.items() if not k.startswith("__")},
                              init_kwargs={"wandb": {
                                  "name": parse_run_name(),
                                  "id": wandb_run_id, 
                                  "resume": resume_flag
                              }})
    
    if accelerator.is_main_process:
        wandb_run_id = accelerator.get_tracker("wandb").run.id

    print(f"Loading DiT & VAE...")
    model = SingleStreamDiT(
        in_channels=Config.in_channels,
        patch_size=Config.patch_size,
        hidden_size=Config.hidden_size,
        depth=Config.depth,
        num_heads=Config.num_heads,
        text_embed_dim=Config.text_embed_dim,
        gradient_checkpointing=Config.gradient_checkpointing,
        refiner_depth=Config.refiner_depth,
        max_token_length=Config.max_token_length,
        dropout=Config.model_dropout,
        rope_base=Config.rope_base
    ).to(Config.device, Config.dtype)
    
    vae = load_vae()
    
    model.initialize_weights()
    print_model_parameters(model)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(Config.ema_decay))
    ema_model.eval()
    ema_model.requires_grad_(False)
    
    full_dataset = TextImageDataset()
    train_idx_set = set(range(len(full_dataset)))
    train_buckets = {res: [i for i in idxs if i in train_idx_set] for res, idxs in full_dataset.buckets.items() if any(i in train_idx_set for i in idxs)}
    if accelerator.num_processes > 1:
        new_buckets = {}
        for res, indices in train_buckets.items():
            remainder = len(indices) % accelerator.num_processes
            if remainder > 0:
                padding_size = accelerator.num_processes - remainder
                indices = indices + indices[:padding_size]
                
            sharded_indices = indices[accelerator.process_index::accelerator.num_processes]
            if sharded_indices:
                new_buckets[res] = sharded_indices
        train_buckets = new_buckets
    train_loader = DataLoader(full_dataset, 
                              batch_sampler=BucketBatchSampler(train_buckets, batch_size=Config.batch_size), 
                              num_workers=Config.num_workers)
    
    steps_per_epoch = math.ceil(len(train_loader) / Config.accum_steps)
    total_steps = steps_per_epoch * Config.epochs
    warmup_steps = int(total_steps * Config.optimizer_warmup)

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    start_epoch = 0
    global_step = 0
    
    resumed = False
    if can_attempt_resume(Config.resume_from):
        resolved_path = checkpoint_manager._resolve_path(Config.resume_from)
        if resolved_path is not None:
            opt_ref = None if Config.reset_optimizer else optimizer
            sched_ref = None if Config.reset_optimizer else scheduler
            start_epoch, global_step = checkpoint_manager.load(resolved_path, model, ema_model, opt_ref, sched_ref)
            resumed = True
            if Config.reset_optimizer:
                print(f"Resetting Scheduler for remaining epochs: {Config.epochs - start_epoch}")
                remaining_epochs = Config.epochs - start_epoch
                if remaining_epochs < 1:
                    remaining_epochs = 10
                new_total_steps = steps_per_epoch * remaining_epochs
                new_warmup = int(new_total_steps * Config.optimizer_warmup)
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=new_warmup, 
                                                            num_training_steps=new_total_steps)
        else:
            print(f"Could not resolve resume path: {Config.resume_from}")
            
            remaining_epochs = Config.epochs - start_epoch
            if remaining_epochs < 1:
                print("Warning: No epochs remaining. Increasing limit by 10.")
                Config.epochs += 10
                remaining_epochs = 10
                
            new_total_steps = steps_per_epoch * remaining_epochs
            new_warmup = int(new_total_steps * Config.optimizer_warmup)
            
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=new_warmup, num_training_steps=new_total_steps)

    if sys.platform.startswith('linux') and Config.compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Successfully compiled model.")
        except Exception as e: 
            print(f"Compilation bypassed: {e}")

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    unwrapped_model = accelerator.unwrap_model(model)
    if not resumed:
        if hasattr(ema_model, 'module'):
            ema_model.module.load_state_dict(unwrapped_model.state_dict())
        else:
            ema_model.load_state_dict(unwrapped_model.state_dict())

    logger = CSVLogger(Config.log_file, resume=(Config.resume_from is not None))

    for epoch in range(start_epoch, Config.epochs):
        display_epoch = epoch + 1
        
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
        running_loss = 0.0
        micro_steps = 0
        binned_sums = {"loss_t_noise": 0.0, "loss_t_mid": 0.0, "loss_t_image": 0.0}
        binned_counts = {"loss_t_noise": 0, "loss_t_mid": 0, "loss_t_image": 0}
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                x_t, t, x_1, target, text, text_mask = prepare_batch_and_targets(batch, Config.device, torch.float32,
                                                                                 Config.shift_val)
                
                with accelerator.autocast():
                    loss_batch = calculate_total_loss(model, x_t, t, target, text, text_mask, Config.loss_type)
                    loss = loss_batch.mean()

                accelerator.backward(loss)
                running_loss += loss.item()
                micro_steps += 1
                         
                with torch.no_grad():
                    t_flat = t.view(-1)
                    for mask, key in [
                        (t_flat < 0.33, "loss_t_noise"),
                        ((t_flat >= 0.33) & (t_flat < 0.66), "loss_t_mid"),
                        (t_flat >= 0.66, "loss_t_image")
                    ]:
                        if mask.any():
                            binned_sums[key] += loss_batch[mask].mean().item()
                            binned_counts[key] += 1
                       
                if accelerator.sync_gradients:
                    global_step += 1
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    ema_model.update_parameters(accelerator.unwrap_model(model))
                                        
                    if global_step % LOG_EVERY_STEPS == 0:
                        lr_current = optimizer.param_groups[0]['lr']
                        curr_loss = running_loss / micro_steps
                        
                        pbar.set_description(f"Epoch {epoch}|Step {global_step}|Loss {curr_loss:.3f}|Loss {Config.loss_type}|LR {lr_current:.6f}")
                        
                        if accelerator.is_main_process:
                            logger.log(epoch, global_step, curr_loss, lr_current)
                        
                        log_dict = {
                            "loss": curr_loss,
                            "lr": lr_current, 
                            "epoch": epoch
                        }
                        for k in binned_sums.keys():
                            if binned_counts[k] > 0:
                                log_dict[k] = binned_sums[k] / binned_counts[k]
                                
                        accelerator.log(log_dict, step=global_step)
                        
                        running_loss = 0.0
                        micro_steps = 0
                        binned_sums = {k: 0.0 for k in binned_sums}
                        binned_counts = {k: 0 for k in binned_counts}

        if display_epoch > 0 and display_epoch % Config.validate_every == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            validate(accelerator, unwrapped_model, vae, display_epoch, global_step, is_ema=False)
            validate(accelerator, ema_model.module, vae, display_epoch, global_step, is_ema=True)
            accelerator.wait_for_everyone()
            
        if accelerator.is_main_process and display_epoch > 0 and display_epoch % Config.save_every == 0:
            unwrapped = accelerator.unwrap_model(model)
            checkpoint_manager.save(display_epoch, global_step, unwrapped, ema_model, optimizer, scheduler, wandb_run_id)

    if accelerator.is_main_process:
            print("Training complete. Saving final checkpoint...")
            unwrapped = accelerator.unwrap_model(model)
            checkpoint_manager.save(Config.epochs, global_step, unwrapped, ema_model, optimizer, scheduler, wandb_run_id, is_final=True)

    accelerator.end_training()
           
if __name__ == "__main__":
    start_time = time.time()
    train()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")