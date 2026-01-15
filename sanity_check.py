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
from train import get_gate_stats
from config import Config
from latents import decode_latents_to_image
from samplers import run_sampling_pipeline
from losses import calculate_total_loss, prepare_batch_and_targets
import wandb

DEVICE = "cuda"

STEPS = 1000
LEARNING_RATE = 4e-4
SAMPLE_EVERY = 200
SAMPLE_STEPS = 50
ENABLE_RK4 = False
GATE_LEARNING_RATE_FACTOR = 1

RUN_NAME = f"Sanity_FFT_{Config.fourier_stack_depth}"

USE_SELF_EVAL = False 

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

def sanity():
    wandb.init(project=Config.project_name + "_sanity", name=RUN_NAME, 
               config={"lr": LEARNING_RATE,
                       "fourier_depth": Config.fourier_stack_depth,
                       "shift": Config.shift_val,
                       "loss_type": Config.loss_type
                       })
    
    if not os.path.exists(Config.target_file):
        print(f"Error: {Config.target_file} not found.")
        return
    
    print(f"Loading {Config.target_file}...")
    data = torch.load(Config.target_file)
    
    latents = data["latents"].unsqueeze(0).to(DEVICE, Config.dtype)
    
    text = data["text_embeds"].unsqueeze(0).to(DEVICE, Config.dtype)
    h, w = data["height"], data["width"]
    
    print(f"Target Resolution: {w}x{h}")
    print(f"RK4 Enabled: {ENABLE_RK4}")
    
    model = SingleStreamDiT(in_channels=Config.in_channels, gradient_checkpointing=False).to(DEVICE, Config.dtype)
    
    model.initialize_weights() 
    
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(Config.ema_decay))
    
    param_base = []
    param_fourier_gates = []
    
    for name, param in model.named_parameters():
        if 'fourier_filter.gate' in name:
            param_fourier_gates.append(param)
        else:
            param_base.append(param)
            
    GATE_LEARNING_RATE = LEARNING_RATE * GATE_LEARNING_RATE_FACTOR     
    optimizer_grouped_parameters = [
        {'params': param_base, 'lr': LEARNING_RATE, 'weight_decay': Config.weight_decay},
        {'params': param_fourier_gates, 'lr': GATE_LEARNING_RATE, 'weight_decay': 0.0},
    ]
    optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=LEARNING_RATE)  
    
    warmup_steps = int(STEPS * Config.optimizer_warmup)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=STEPS)
    
    print(f"Loading VAE: {Config.vae_id}...")
    vae = AutoencoderKL.from_pretrained(Config.vae_id).to(DEVICE)

    def validate(step_count):
        model.eval()
        with torch.no_grad():
            torch_generator = torch.Generator(device=DEVICE).manual_seed(42)
            initial_noise = torch.randn(1, 16, h // Config.vae_downsample_factor, w // Config.vae_downsample_factor, 
                                        generator=torch_generator, device=DEVICE, dtype=Config.dtype)
   
            uncond_embeds = torch.zeros_like(text)
            combined_text_embeds = torch.cat([uncond_embeds, text], dim=0)       
            x_euler = initial_noise.clone()
            x_rk4 = None
            
            with torch.autocast(device_type=DEVICE, dtype=Config.dtype):
                x_euler = run_sampling_pipeline(model=model, initial_noise=x_euler, steps=SAMPLE_STEPS, 
                                                combined_text_embeds=combined_text_embeds, cfg=1.0, 
                                                sampler_type="euler", shift_val=Config.shift_val)
                if ENABLE_RK4:
                    x_rk4 = run_sampling_pipeline(model=model, initial_noise=initial_noise.clone(), steps=SAMPLE_STEPS,
                                                  combined_text_embeds=combined_text_embeds, cfg=1.0, 
                                                  sampler_type="rk4", shift_val=Config.shift_val)

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
            "text_embeds": text 
        }
        x_t, t, x_1, target, text_for_model = prepare_batch_and_targets(batch_data, DEVICE, Config.dtype, Config.shift_val, Config.offset_noise)
        
        with torch.autocast(device_type=DEVICE, dtype=Config.dtype):
            if USE_SELF_EVAL and step > (STEPS * Config.start_self_eval_at):
                 with torch.no_grad():
                    loss = calculate_total_loss(model, ema_model, x_t, t, x_1, target, text_for_model, step, STEPS,
                                                USE_SELF_EVAL, Config.start_self_eval_at, Config.self_eval_lambda, Config.fal_lambda, 
                                                Config.fcl_lambda, Config.loss_type, 1)
            else:
                loss = calculate_total_loss(model, ema_model, x_t, t, x_1, target, text_for_model, step, STEPS, 
                                            USE_SELF_EVAL, Config.start_self_eval_at, Config.self_eval_lambda, Config.fal_lambda, 
                                            Config.fcl_lambda, Config.loss_type, 1)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update_parameters(model)
        
        lr_current = optimizer.param_groups[0]['lr']
        avg_gate, min_gate, max_gate = get_gate_stats(model) 
        self_eval_status = "On" if USE_SELF_EVAL and step > (STEPS * Config.start_self_eval_at) else "Off"     
        
        wandb.log({"loss": loss.item(),
                   "lr": lr_current,
                   "gate_avg": avg_gate,
                   "gate_min": min_gate,
                   "gate_max": max_gate}, 
                  step=step)
        
        pbar.set_description(f"Step {step}|Loss: {loss.item():.3f}|LR: {lr_current:.6f}|Gate(avg-min-max): {avg_gate:.3f}[{min_gate:.3f}/{max_gate:.3f}]|Self-E: {self_eval_status}|")
                
        if step > 0 and (step % SAMPLE_EVERY == 0 or step == STEPS - 1):
            validate(step)
            
    wandb.finish()

if __name__ == "__main__":
    start_time = time.time()
    sanity()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")