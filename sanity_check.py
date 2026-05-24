import os
import torch
import time
import bitsandbytes as bnb
from tqdm import tqdm
from model import SingleStreamDiT
from PIL import Image, ImageDraw, ImageFont
from transformers import get_cosine_schedule_with_warmup 
from model_loader import load_vae
from config import Config
from latents import decode_latents_to_image, load_repa_target
from samplers import run_sampling_pipeline
from losses import calculate_total_loss, prepare_batch_and_targets
import wandb
from utilities import parse_run_name, print_model_parameters

DEVICE = "cuda"

STEPS = 1000
LEARNING_RATE = 4e-4
SAMPLE_EVERY = 200
SAMPLE_STEPS = 50
ENABLE_RK4 = False

# =====================================================================
# CONFIGURATION: MANUAL FILE OVERRIDES
# Set these to specific filenames in your cache directory to test 
# Set to None to let the script automatically grab the first two files.
# =====================================================================
MANUAL_FILE_A = "1349.pt"
MANUAL_FILE_B = "34953.pt"
# =====================================================================

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True) 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

def sanity():
    wandb.init(project=Config.project_name + "_sanity", name=parse_run_name(LEARNING_RATE) + "_4-image", 
               config={"lr": LEARNING_RATE,                       
                       "shift": Config.shift_val,
                       "loss_type": Config.loss_type
                       })
    
    if MANUAL_FILE_A is not None and MANUAL_FILE_B is not None:
        paired_files = [MANUAL_FILE_A, MANUAL_FILE_B]
    else:
        cache_files = sorted([f for f in os.listdir(Config.cache_dir) if f.endswith('.pt')])
        if len(cache_files) < 2:
            print(f"Error: Need at least 2 cached .pt files in {Config.cache_dir} to run a multi-concept check.")
            return
        paired_files = cache_files[:2]
    
    file_a = os.path.join(Config.cache_dir, paired_files[0])
    file_b = os.path.join(Config.cache_dir, paired_files[1])
    
    if not os.path.exists(file_a):
        print(f"Error: Specific file A not found at path: {file_a}")
        return
    if not os.path.exists(file_b):
        print(f"Error: Specific file B not found at path: {file_b}")
        return
    
    print(f"Selected paired files for sanity check:")
    print(f"  File A: {paired_files[0]}")
    print(f"  File B: {paired_files[1]}")
    
    data_a = torch.load(file_a)
    data_b = torch.load(file_b)
    
    h_a, w_a = data_a["height"], data_a["width"]
    h_b, w_b = data_b["height"], data_b["width"]
    
    latents_a = data_a["latents"].unsqueeze(0).to(DEVICE, Config.dtype)
    latents_b = data_b["latents"].unsqueeze(0).to(DEVICE, Config.dtype)
    
    repa_a = load_repa_target(data_a, Config.dtype).unsqueeze(0).to(DEVICE)
    repa_b = load_repa_target(data_b, Config.dtype).unsqueeze(0).to(DEVICE)
    
    def get_full_text_lists(data):
        if "text_embeds_list" in data:
            embeds_list = data["text_embeds_list"].to(Config.device, Config.dtype)
            mask_list = data["attention_mask_list"].to(Config.device)
        else:
            embeds_list = data["text_embeds"].unsqueeze(0).to(Config.device, Config.dtype)
            mask_list = data["attention_mask"].unsqueeze(0).to(Config.device)
        return embeds_list, mask_list

    embeds_list_a, mask_list_a = get_full_text_lists(data_a)
    embeds_list_b, mask_list_b = get_full_text_lists(data_b)
    
    can_batch = (h_a == h_b) and (w_a == w_b)
    if can_batch:
        print(f"Resolutions match ({w_a}x{h_a}). Training in parallel (Batch Size = 2).")
        latents_batch = torch.cat([latents_a, latents_b], dim=0)
        repa_batch = torch.cat([repa_a, repa_b], dim=0)
    else:
        print(f"Resolutions mismatch ({w_a}x{h_a} vs {w_b}x{h_b}). Training sequentially via alternating steps.")

    model = SingleStreamDiT(
        in_channels=Config.in_channels,
        patch_size=Config.patch_size,
        hidden_size=Config.hidden_size,
        depth=Config.depth,
        num_heads=Config.num_heads,
        text_embed_dim=Config.text_embed_dim,
        refiner_depth=Config.refiner_depth,
    ).to(Config.device, Config.dtype)
    
    model.initialize_weights() 
    print_model_parameters(model)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name.lower() or "bias" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
            
    optim_groups = [
        {"params": decay, "weight_decay": Config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
    optimizer = bnb.optim.AdamW8bit(optim_groups, lr=LEARNING_RATE)
    
    warmup_steps = int(STEPS * Config.optimizer_warmup)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=STEPS)
    
    vae = load_vae()

    def generate_sample(embeds, mask, h, w, torch_generator):
        initial_noise = torch.randn(1, Config.in_channels, h // Config.vae_downsample_factor,
                                      w // Config.vae_downsample_factor, generator=torch_generator, device=DEVICE,
                                      dtype=Config.dtype)
        uncond_embeds = torch.zeros_like(embeds)
        uncond_mask = torch.ones_like(mask)
        combined_text_embeds = torch.cat([uncond_embeds, embeds], dim=0)
        combined_mask = torch.cat([uncond_mask, mask], dim=0)
        
        with torch.autocast(device_type=DEVICE, dtype=Config.dtype):
            x_euler = run_sampling_pipeline(model=model, initial_noise=initial_noise, steps=SAMPLE_STEPS,
                                              combined_text_embeds=combined_text_embeds, cfg=1.0,
                                              sampler_type="euler", scheduler_type="uniform",
                                              shift_val=Config.shift_val, text_mask=combined_mask)
        return decode_latents_to_image(vae_model=vae, latents=x_euler, device=DEVICE)

    def validate(step_count):
        with torch.no_grad():
            torch_generator = torch.Generator(device=DEVICE).manual_seed(42)
            label_height = 40
            
            images_to_render = []
            labels_to_render = []
            
            num_caps_a = min(2, embeds_list_a.shape[0])
            for i in range(num_caps_a):
                img = generate_sample(embeds_list_a[i].unsqueeze(0), mask_list_a[i].unsqueeze(0), h_a, w_a, torch_generator)
                images_to_render.append(img)
                labels_to_render.append(f"{paired_files[0]} (Description {i+1})")
                
            num_caps_b = min(2, embeds_list_b.shape[0])
            for i in range(num_caps_b):
                img = generate_sample(embeds_list_b[i].unsqueeze(0), mask_list_b[i].unsqueeze(0), h_b, w_b, torch_generator)
                images_to_render.append(img)
                labels_to_render.append(f"{paired_files[1]} (Description {i+1})")

            total_w = sum(img.size[0] for img in images_to_render)
            max_h = max(img.size[1] for img in images_to_render)
            
            canvas = Image.new("RGB", (total_w, max_h + label_height), color=(0, 0, 0))
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            current_x = 0
            draw = ImageDraw.Draw(canvas)
            for img, label in zip(images_to_render, labels_to_render):
                canvas.paste(img, (current_x, label_height))
                draw.text((current_x + (img.size[0] // 2), label_height // 2), label, fill=(255, 255, 255), font=font, anchor="mm")
                current_x += img.size[0]
                
            filename = f"sanity_match_step_{step_count:04d}.png"
            canvas.save(filename)
            wandb.log({"sanity_sample": wandb.Image(canvas)}, step=step_count)
             
        model.train()
        
    print(f"Starting Semantic Multicaption Overfit (Shift={Config.shift_val})...")
    pbar = tqdm(range(STEPS))
    
    for step in pbar:
        idx_a = torch.randint(0, embeds_list_a.shape[0], (1,)).item()
        idx_b = torch.randint(0, embeds_list_b.shape[0], (1,)).item()
        
        embeds_a = embeds_list_a[idx_a].unsqueeze(0)
        mask_a = mask_list_a[idx_a].unsqueeze(0)
        
        embeds_b = embeds_list_b[idx_b].unsqueeze(0)
        mask_b = mask_list_b[idx_b].unsqueeze(0)

        if can_batch:
            # ⚡ Keep dynamic text padding
            max_len = max(embeds_a.shape[1], embeds_b.shape[1])
            
            if embeds_a.shape[1] < max_len:
                pad_len = max_len - embeds_a.shape[1]
                embeds_a = torch.cat([embeds_a, torch.zeros(1, pad_len, embeds_a.shape[2], device=DEVICE, dtype=Config.dtype)], dim=1)
                mask_a = torch.cat([mask_a, torch.zeros(1, pad_len, device=DEVICE, dtype=torch.bool)], dim=1)
                
            if embeds_b.shape[1] < max_len:
                pad_len = max_len - embeds_b.shape[1]
                embeds_b = torch.cat([embeds_b, torch.zeros(1, pad_len, embeds_b.shape[2], device=DEVICE, dtype=Config.dtype)], dim=1)
                mask_b = torch.cat([mask_b, torch.zeros(1, pad_len, device=DEVICE, dtype=torch.bool)], dim=1)

            text_embeds_batch = torch.cat([embeds_a, embeds_b], dim=0)
            text_mask_batch = torch.cat([mask_a, mask_b], dim=0)
            
            batch_data = {
                "latents": latents_batch, 
                "repa_target": repa_batch,
                "text_embeds": text_embeds_batch,
                "text_mask": text_mask_batch
            }
        else:
            if step % 2 == 0:
                batch_data = {"latents": latents_a, "repa_target": repa_a, "text_embeds": embeds_a, "text_mask": mask_a}
            else:
                batch_data = {"latents": latents_b, "repa_target": repa_b, "text_embeds": embeds_b, "text_mask": mask_b}
                
        x_t, t, x_1, target, text_for_model, mask_for_model = prepare_batch_and_targets(batch_data, 
                                                                                        DEVICE, 
                                                                                        Config.dtype, 
                                                                                        Config.shift_val)
                
        repa_target_batch = batch_data.get("repa_target", None)
        
        with torch.autocast(device_type=DEVICE, dtype=Config.dtype):
            if repa_target_batch is not None and repa_target_batch.dim() > 1:
                loss_batch, base_loss_batch, repa_loss_batch = calculate_total_loss(model, x_t, t, target, 
                                                                                    text_for_model, mask_for_model,
                                                                                    Config.loss_type, 
                                                                                    repa_target=repa_target_batch,
                                                                                    repa_lambda=Config.repa_lambda)
                loss = loss_batch.mean()
                base_loss_for_bin = base_loss_batch
                repa_loss_val = repa_loss_batch.mean().item()
            else:
                base_loss_batch = calculate_total_loss(model, x_t, t, target, text_for_model, mask_for_model, Config.loss_type)
                loss = base_loss_batch.mean()
                base_loss_for_bin = base_loss_batch
                repa_loss_val = 0.0
                
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        binned_logs = {}
        with torch.no_grad():
            t_flat = t.view(-1)
            if (t_flat < 0.33).any(): binned_logs["loss_t_noise"] = base_loss_for_bin[t_flat < 0.33].mean().item()
            if ((t_flat >= 0.33) & (t_flat < 0.66)).any(): binned_logs["loss_t_mid"] = base_loss_for_bin[(t_flat >= 0.33) & (t_flat < 0.66)].mean().item()
            if (t_flat >= 0.66).any(): binned_logs["loss_t_image"] = base_loss_for_bin[t_flat >= 0.66].mean().item()
        
        lr_current = optimizer.param_groups[0]['lr']
        log_dict = {"loss": loss.item(), "loss_repa": repa_loss_val, "lr": lr_current}
        log_dict.update(binned_logs)
        wandb.log(log_dict, step=step)
        
        pbar.set_description(f"Step {step}|Loss {loss.item():.3f}|REPA {repa_loss_val:.3f}|Loss {Config.loss_type}|LR {lr_current:.6f}|")
                            
        if step > 0 and (step % SAMPLE_EVERY == 0 or step == STEPS - 1):
            validate(step)
            
    wandb.finish()

if __name__ == "__main__":
    start_time = time.time()
    sanity()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")