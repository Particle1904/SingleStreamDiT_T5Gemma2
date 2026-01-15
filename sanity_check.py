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

TARGET_FILE = Config.target_file
DEVICE = Config.device
DTYPE = Config.dtype

steps = 1000
learning_rate = 4e-4
sample_every = 200
sample_steps = 50
enable_rk4 = False
gate_learning_rate_factor = 20

LOSS_TYPE = Config.loss_type
VAE_ID = Config.vae_id
SHIFT_VAL = Config.shift_val
EMA_DECAY = Config.ema_decay
WEIGHT_DECAY = Config.weight_decay
IN_CHANNELS = Config.in_channels
OPTIMIZER_WARMUP = Config.optimizer_warmup
OFFSET_NOISE = Config.offset_noise

VAE_DOWNSAMPLE_FACTOR = Config.vae_downsample_factor

USE_SELF_EVAL = False 
SELF_EVAL_LAMBDA = Config.self_eval_lambda  
START_SELF_EVAL_AT = Config.start_self_eval_at
FAL_LAMBDA = Config.fal_lambda
FCL_LAMBDA = Config.fcl_lambda

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def sanity():
    if not os.path.exists(TARGET_FILE):
        print(f"Error: {TARGET_FILE} not found.")
        return
    
    print(f"Loading {TARGET_FILE}...")
    data = torch.load(TARGET_FILE)
    
    latents = data["latents"].unsqueeze(0).to(DEVICE, DTYPE)
    
    text = data["text_embeds"].unsqueeze(0).to(DEVICE, DTYPE)
    h, w = data["height"], data["width"]
    
    print(f"Target Resolution: {w}x{h}")
    print(f"RK4 Enabled: {enable_rk4}")
    
    model = SingleStreamDiT(in_channels=IN_CHANNELS, gradient_checkpointing=False).to(DEVICE, DTYPE)
    
    model.initialize_weights() 
    
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    
    #optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)   
    param_base = []
    param_fourier_gates = []
    
    for name, param in model.named_parameters():
        if 'fourier_filter.gate' in name:
            param_fourier_gates.append(param)
        else:
            param_base.append(param)
            
    GATE_LEARNING_RATE = learning_rate * gate_learning_rate_factor     
    optimizer_grouped_parameters = [
        {'params': param_base, 'lr': learning_rate, 'weight_decay': WEIGHT_DECAY},
        {'params': param_fourier_gates, 'lr': GATE_LEARNING_RATE, 'weight_decay': 0.0},
    ]
    optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=learning_rate)  
    
    warmup_steps = int(steps * OPTIMIZER_WARMUP)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)
    
    print(f"Loading VAE: {VAE_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE)

    def validate(step_count):
        model.eval()
        with torch.no_grad():
            torch_generator = torch.Generator(device=DEVICE).manual_seed(42)
            initial_noise = torch.randn(1, 16, h // VAE_DOWNSAMPLE_FACTOR, w // VAE_DOWNSAMPLE_FACTOR, 
                                        generator=torch_generator, device=DEVICE, dtype=DTYPE)
   
            uncond_embeds = torch.zeros_like(text)
            combined_text_embeds = torch.cat([uncond_embeds, text], dim=0)       
            x_euler = initial_noise.clone()
            x_rk4 = None
            
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                x_euler = run_sampling_pipeline(model=model, initial_noise=x_euler, steps=sample_steps, 
                                                combined_text_embeds=combined_text_embeds, cfg=1.0, 
                                                sampler_type="euler", shift_val=SHIFT_VAL)
                if enable_rk4:
                    x_rk4 = run_sampling_pipeline(model=model, initial_noise=initial_noise.clone(), steps=sample_steps,
                                                  combined_text_embeds=combined_text_embeds, cfg=1.0, 
                                                  sampler_type="rk4", shift_val=SHIFT_VAL)

            img_pil_euler = decode_latents_to_image(vae_model=vae, latents=x_euler, device=DEVICE)
            img_list = [img_pil_euler]
            
            if enable_rk4:
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
            
            if enable_rk4:
                half_w = w_total // 2
                draw.text((half_w // 2, label_y), "Euler", fill=(255, 255, 255), font=font, anchor="mm")
                draw.text((half_w + half_w // 2, label_y), "RK4", fill=(255, 255, 255), font=font, anchor="mm")
            else:
                draw.text((w_total // 2, label_y), "Euler", fill=(255, 255, 255), font=font, anchor="mm")

            filename = f"sanity_match_step_{step_count:04d}.png"
            canvas.save(filename)

        model.train()
        
    print(f"Starting Overfit (Shift={SHIFT_VAL}, SelfEval={USE_SELF_EVAL})...")
    pbar = tqdm(range(steps))
    
    for step in pbar:        
        batch_data = {
            "latents": latents, 
            "text_embeds": text 
        }
        x_t, t, x_1, target, text_for_model = prepare_batch_and_targets(batch_data, DEVICE, DTYPE, SHIFT_VAL, OFFSET_NOISE)
        
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            if USE_SELF_EVAL and step > (steps * START_SELF_EVAL_AT):
                 with torch.no_grad():
                    loss = calculate_total_loss(model, ema_model, x_t, t, x_1, target, text_for_model, step, steps,
                                                USE_SELF_EVAL, START_SELF_EVAL_AT, SELF_EVAL_LAMBDA, FAL_LAMBDA, 
                                                FCL_LAMBDA, LOSS_TYPE, 1)
            else:
                loss = calculate_total_loss(model, ema_model, x_t, t, x_1, target, text_for_model, step, steps, 
                                            USE_SELF_EVAL, START_SELF_EVAL_AT, SELF_EVAL_LAMBDA, FAL_LAMBDA, 
                                            FCL_LAMBDA, LOSS_TYPE, 1)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update_parameters(model)
        
        lr_current = optimizer.param_groups[0]['lr']
        avg_gate, min_gate, max_gate = get_gate_stats(model) 
        self_eval_status = "On" if USE_SELF_EVAL and step > (steps * START_SELF_EVAL_AT) else "Off"     
        pbar.set_description(f"Step {step}|Loss: {loss.item():.3f}|LR: {lr_current:.6f}|Gate(avg-min-max): {avg_gate:.3f}[{min_gate:.3f}/{max_gate:.3f}]|Self-E: {self_eval_status}|")
                
        if step > 0 and (step % sample_every == 0 or step == steps - 1):
            validate(step)

if __name__ == "__main__":
    start_time = time.time()
    sanity()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")