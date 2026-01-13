import os
import torch
import time
import torch.nn.functional as F
import bitsandbytes as bnb
from tqdm import tqdm
from model import SingleStreamDiTV2
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import get_cosine_schedule_with_warmup 
from train import get_average_gate_value
from config import Config
from latents import *
from samplers import *

TARGET_FILE = Config.target_file
DEVICE = Config.device
DTYPE = Config.dtype

steps = 1500
learning_rate = 4e-4
sample_every = 200
sample_steps = 50
enable_rk4 = False

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
    
    model = SingleStreamDiTV2(in_channels=IN_CHANNELS, gradient_checkpointing=False).to(DEVICE, DTYPE)
    
    model.initialize_weights() 
    
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)    
    
    warmup_steps = int(steps * OPTIMIZER_WARMUP)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)
    
    print(f"Loading VAE: {VAE_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE)

    def generate_image(step_count):
        model.eval()
        with torch.no_grad():
            torch_generator = torch.Generator(device=DEVICE).manual_seed(42)
            x0 = torch.randn(1, 16, h // VAE_DOWNSAMPLE_FACTOR, w // VAE_DOWNSAMPLE_FACTOR, generator=torch_generator, device=DEVICE, dtype=DTYPE)

            x_euler = x0.clone()
            x_rk4 = None
            if enable_rk4:
                x_rk4 = x0.clone()

            dt = 1.0 / sample_steps
            combined_text = torch.cat([torch.zeros_like(text), text], dim=0)

            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                for i in range(sample_steps):
                    t_linear = torch.tensor([i / sample_steps], device=DEVICE, dtype=DTYPE)
                    t = get_1d_shifted_time(t_linear, SHIFT_VAL)
                    x_euler = euler_step(model, x_euler, t, dt, combined_text, 1.0)
                    
                    if enable_rk4:
                        t_mid_linear = torch.tensor([(i + 0.5) / sample_steps], device=DEVICE, dtype=DTYPE)
                        t_mid = get_1d_shifted_time(t_mid_linear, SHIFT_VAL)
                        x_rk4 = rk4_step(model, x_rk4, t, dt, combined_text, 1.0, t_mid)

            latents_euler = prepare_latents_for_decode(x_euler, clamp=False)
            
            with torch.autocast(DEVICE, enabled=False):
                img_euler = vae.decode(latents_euler.float()).sample
                img_euler = (img_euler / 2 + 0.5).clamp(0, 1)                
                img_list = [img_euler]
                
                if enable_rk4:
                    latents_rk4 = prepare_latents_for_decode(x_rk4, clamp=False)
                    img_rk4 = vae.decode(latents_rk4.float()).sample
                    img_rk4 = (img_rk4 / 2 + 0.5).clamp(0, 1)
                    img_list.append(img_rk4)

            img = torch.cat(img_list, dim=3)
            img = img.cpu().permute(0, 2, 3, 1).float().numpy()
            img = (img * 255).round().astype("uint8")

            img_pil = Image.fromarray(img[0])

            label_height = 40
            w_total, h_img = img_pil.size
            canvas = Image.new("RGB", (w_total, h_img + label_height), color=(0, 0, 0))
            canvas.paste(img_pil, (0, label_height))

            draw = ImageDraw.Draw(canvas)

            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            label_y = label_height // 2
            
            if enable_rk4:
                # Draw both labels
                half_w = w_total // 2
                draw.text((half_w // 2, label_y), "Euler", fill=(255, 255, 255), font=font, anchor="mm")
                draw.text((half_w + half_w // 2, label_y), "RK4", fill=(255, 255, 255), font=font, anchor="mm")
            else:
                # Draw single label centered
                draw.text((w_total // 2, label_y), "Euler", fill=(255, 255, 255), font=font, anchor="mm")

            filename = f"sanity_match_step_{step_count:04d}.png"
            canvas.save(filename)

        model.train()
        
    print(f"Starting Overfit (Shift={SHIFT_VAL}, SelfEval={USE_SELF_EVAL})...")
    pbar = tqdm(range(steps))
    
    for step in pbar:        
        x_1 = latents
        
        u = torch.rand(x_1.shape[0], device=DEVICE, dtype=DTYPE)
        t = (u * SHIFT_VAL) / (1 + (SHIFT_VAL - 1) * u)
        
        x_0 = torch.randn_like(x_1)
        x_0 = x_0 + OFFSET_NOISE * torch.randn(x_1.shape[0], x_1.shape[1], 1, 1, device=DEVICE, dtype=DTYPE)
        
        x_t = (1 - t.view(-1,1,1,1)) * x_0 + t.view(-1,1,1,1) * x_1
        
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            v_pred = model(x_t, t, text)
            
            if USE_SELF_EVAL and step > (steps * START_SELF_EVAL_AT):
                x_hat_1 = euler_to_1(x_t, t, v_pred)
                
                with torch.no_grad():
                    s = t + torch.rand_like(t) * (1.0 - t)
                    
                    noise_s = torch.randn_like(x_hat_1)
                    x_hat_s = (1.0 - s.view(-1, 1, 1, 1)) * noise_s + s.view(-1, 1, 1, 1) * x_hat_1
                    
                    teacher_net = ema_model.module
                    text_uncond = torch.zeros_like(text)
                    combined_text = torch.cat([text_uncond, text], dim=0)

                    x_guided = cfg_guided_position(model=teacher_net, x=x_hat_s, t=s, text_embeds=combined_text, cfg=1.0)

                    x_self = x_hat_1 + (x_guided - x_hat_s)
                    
                    lambd_weight = ((1.0 - t) / (t + 1e-4)) - ((1.0 - s) / (s + 1e-4))
                    lambd_weight = lambd_weight.view(-1, 1, 1, 1).clamp(0, 10) * SELF_EVAL_LAMBDA
                    
                    target_raw = x_1 + lambd_weight * x_self
                    
                    norm_clean = torch.linalg.vector_norm(x_1, dim=(1, 2, 3), keepdim=True)
                    norm_target = torch.linalg.vector_norm(target_raw, dim=(1, 2, 3), keepdim=True)
                    norm_factor = norm_clean / (norm_target + 1e-6)
                    x_renorm = target_raw * norm_factor

                loss = F.mse_loss(x_hat_1, x_renorm)
                status_msg = "Self-Eval"
            else:
                target = x_1 - x_0
                loss = F.mse_loss(v_pred, target)
                status_msg = "Standard"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update_parameters(model)
        
        lr_curr = optimizer.param_groups[0]['lr']
        avg_gate = get_average_gate_value(model)
        pbar.set_description(f"[{status_msg}] | Loss: {loss.item():.4f} | F-Gate: {avg_gate:.4f} | LR: {lr_curr:.6f}")

        if step > 0 and (step % sample_every == 0 or step == steps - 1):
            generate_image(step)

    #print("\nSanity Check Complete. Saving weights...")
    #torch.save(model.state_dict(), "sanity_overfit_shift3.pt")

if __name__ == "__main__":
    start_time = time.time()
    sanity()
    final_time = time.time() - start_time
    print(f"Total time in minutes: {final_time / 60:.2f}")