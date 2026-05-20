import os
import torch
import time
import bitsandbytes as bnb
from tqdm import tqdm
from model import SingleStreamDiT
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import get_cosine_schedule_with_warmup 
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
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

USE_SELF_EVAL = False 

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

def sanity():
    wandb.init(project=Config.project_name + "_sanity", name=parse_run_name(LEARNING_RATE), 
               config={"lr": LEARNING_RATE,                       
                       "shift": Config.shift_val,
                       "loss_type": Config.loss_type
                       })
    
    if not os.path.exists(Config.target_file):
        print(f"Error: {Config.target_file} not found.")
        return
    
    print(f"Loading {Config.target_file}...")
    data = torch.load(Config.target_file)    
    latents = data["latents"].unsqueeze(0).to(DEVICE, Config.dtype)
    
    if "text_embeds_list" in data:
        text_embeds = data["text_embeds_list"][0].unsqueeze(0).to(Config.device, Config.dtype)
        text_mask = data["attention_mask_list"][0].unsqueeze(0).to(Config.device)
    else:
        text_embeds = data["text_embeds"].unsqueeze(0).to(Config.device, Config.dtype)
        text_mask = data["attention_mask"].unsqueeze(0).to(Config.device)
    
    h, w = data["height"], data["width"]
    
    print(f"Target Resolution: {w}x{h}")
    print(f"RK4 Enabled: {ENABLE_RK4}")
    
    model = SingleStreamDiT(
        in_channels=Config.in_channels,
        patch_size=Config.patch_size,
        hidden_size=Config.hidden_size,
        depth=Config.depth,
        num_heads=Config.num_heads,
        text_embed_dim=Config.text_embed_dim,
        gradient_checkpointing=False,
        refiner_depth=Config.refiner_depth,
        max_token_length=Config.max_token_length,
        dropout=Config.model_dropout,
        rope_base=Config.rope_base
    ).to(Config.device, Config.dtype)
    
    model.initialize_weights() 
    print_model_parameters(model)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(Config.ema_decay))
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=Config.weight_decay)
    
    warmup_steps = int(STEPS * Config.optimizer_warmup)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=STEPS)
    
    vae = load_vae()

    def validate(step_count):
        model.eval()
        with torch.no_grad():
            torch_generator = torch.Generator(device=DEVICE).manual_seed(42)
            initial_noise = torch.randn(1, Config.in_channels, h // Config.vae_downsample_factor, 
                                        w // Config.vae_downsample_factor, generator=torch_generator, device=DEVICE, 
                                        dtype=Config.dtype)
   
            uncond_embeds = torch.zeros_like(text_embeds)
            uncond_mask = torch.ones_like(text_mask)
            combined_text_embeds = torch.cat([uncond_embeds, text_embeds], dim=0)
            combined_mask = torch.cat([uncond_mask, text_mask], dim=0)
                 
            x_euler = initial_noise.clone()
            x_rk4 = None
            
            with torch.autocast(device_type=DEVICE, dtype=Config.dtype):
                x_euler = run_sampling_pipeline(model=model, initial_noise=x_euler, steps=SAMPLE_STEPS, 
                                                combined_text_embeds=combined_text_embeds, cfg=1.0, 
                                                sampler_type="euler", scheduler_type="uniform", 
                                                shift_val=Config.shift_val, text_mask=combined_mask)
                if ENABLE_RK4:
                    x_rk4 = run_sampling_pipeline(model=model, initial_noise=initial_noise.clone(), steps=SAMPLE_STEPS,
                                                  combined_text_embeds=combined_text_embeds, cfg=1.0, 
                                                  sampler_type="rk4", scheduler_type="karras",
                                                  shift_val=Config.shift_val, text_mask=combined_mask)

            img_pil_euler = decode_latents_to_image(vae_model=vae, latents=x_euler, device=DEVICE)
            img_list = [img_pil_euler]
            
            if ENABLE_RK4:
                img_pil_rk4 = decode_latents_to_image(vae_model=vae, latents=x_rk4, device=DEVICE)
                img_list.append(img_pil_rk4)

            w_euler, h_img = img_pil_euler.size
            w_total = w_euler * len(img_list)
            label_height = 40
            canvas = Image.new("RGB", (w_total, h_img + label_height), color=(0, 0, 0))
            current_x = 0
            for img_pil in img_list:
                canvas.paste(img_pil, (current_x, label_height))
                current_x += img_pil.size[0]
                
            draw = ImageDraw.Draw(canvas)

            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            label_y = label_height // 2
            
            if ENABLE_RK4:
                half_w = w_total // 2
                draw.text((half_w // 2, label_y), "Euler", fill=(255, 255, 255), font=font, anchor="mm")
                draw.text((half_w + half_w // 2, label_y), "RK4", fill=(255, 255, 255), font=font, anchor="mm")
            else:
                draw.text((w_total // 2, label_y), "Euler", fill=(255, 255, 255), font=font, anchor="mm")

            filename = f"sanity_match_step_{step_count:04d}.png"
            canvas.save(filename)
            wandb.log({"sanity_sample": wandb.Image(canvas)}, step=step_count)
             
        model.train()
        
    print(f"Starting Overfit (Shift={Config.shift_val}, SelfEval={USE_SELF_EVAL})...")
    pbar = tqdm(range(STEPS))
    
    for step in pbar:        
        batch_data = {
            "latents": latents, 
            "text_embeds": text_embeds,
            "text_mask": text_mask
        }
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