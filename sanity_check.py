import os
import torch
import time
import bitsandbytes as bnb
from tqdm import tqdm
from model import SingleStreamDiT
from PIL import Image, ImageDraw, ImageFont
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import get_cosine_schedule_with_warmup 
from model_loader import load_vae
from config import Config
from latents import decode_latents_to_image
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
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(Config.ema_decay))
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=Config.weight_decay)
    
    warmup_steps = int(STEPS * Config.optimizer_warmup)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=STEPS)
    
    vae = load_vae()

    def generate_sample(embeds, mask, h, w, torch_generator):
        initial_noise = torch.randn(1, Config.in_channels, h // Config.vae_downsample_factor, 
                                      w // Config.vae_downsample_factor, generator=torch_generator, device=DEVICE, 
                                      dtype=Config.dtype)
        uncond_embeds = torch.zeros_like(embeds)
        uncond_mask = torch.ones_like(mask)
        comb_embeds = torch.cat([uncond_embeds, embeds], dim=0)
        comb_mask = torch.cat([uncond_mask, mask], dim=0)
        
        with torch.autocast(device_type=DEVICE, dtype=Config.dtype):
            x_euler = run_sampling_pipeline(model=model, initial_noise=initial_noise, steps=SAMPLE_STEPS, 
                                              combined_text_embeds=comb_embeds, cfg=1.0, 
                                              sampler_type="euler", scheduler_type="uniform", 
                                              shift_val=Config.shift_val, text_mask=comb_mask)
        return decode_latents_to_image(vae_model=vae, latents=x_euler, device=DEVICE)

    def validate(step_count):
        model.eval()
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
            text_embeds_batch = torch.cat([embeds_a, embeds_b], dim=0)
            text_mask_batch = torch.cat([mask_a, mask_b], dim=0)
            batch_data = {
                "latents": latents_batch, 
                "text_embeds": text_embeds_batch,
                "text_mask": text_mask_batch
            }
        else:
            if step % 2 == 0:
                batch_data = {"latents": latents_a, "text_embeds": embeds_a, "text_mask": mask_a}
            else:
                batch_data = {"latents": latents_b, "text_embeds": embeds_b, "text_mask": mask_b}
                
        x_t, t, x_1, target, text_for_model, mask_for_model = prepare_batch_and_targets(batch_data, DEVICE, 
                                                                                        Config.dtype, 
                                                                                        Config.shift_val)
        
        with torch.autocast(device_type=DEVICE, dtype=Config.dtype):
            loss_batch = calculate_total_loss(model, x_t, t, target, text_for_model, mask_for_model, 
                                              Config.loss_type)
            loss = loss_batch.mean()
                
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ema_model.update_parameters(model)
        
        binned_logs = {}
        with torch.no_grad():
            t_flat = t.view(-1)
            if (t_flat < 0.33).any(): binned_logs["loss_t_noise"] = loss_batch[t_flat < 0.33].mean().item()
            if ((t_flat >= 0.33) & (t_flat < 0.66)).any(): binned_logs["loss_t_mid"] = loss_batch[(t_flat >= 0.33) & (t_flat < 0.66)].mean().item()
            if (t_flat >= 0.66).any(): binned_logs["loss_t_image"] = loss_batch[t_flat >= 0.66].mean().item()
        
        lr_current = optimizer.param_groups[0]['lr']
        log_dict = {"loss": loss.item(), "lr": lr_current}
        log_dict.update(binned_logs)
        wandb.log(log_dict, step=step)
        
        pbar.set_description(f"Step {step}|Loss {loss.item():.3f}|Loss {Config.loss_type}|LR {lr_current:.6f}|")
                            
        if step > 0 and (step % SAMPLE_EVERY == 0 or step == STEPS - 1):
            validate(step)
            
    wandb.finish()

if __name__ == "__main__":
    start_time = time.time()
    sanity()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")