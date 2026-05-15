import torch
import math
import torch.nn.functional as F
from samplers import cfg_guided_position, predict_x1_from_velocity, get_1d_shifted_time  

def prepare_batch_and_targets(batch, device, dtype, shift_val, offset_noise, loss_type):
    x_1 = batch["latents"].to(device, dtype=dtype)
    text = batch["text_embeds"].to(device, dtype=dtype)
    text_mask = batch["text_mask"].to(device)
    
    u = torch.rand(x_1.shape[0], device=device, dtype=dtype)
    
    t = get_1d_shifted_time(u, shift_val)
    
    x_0 = torch.randn_like(x_1)
    if offset_noise > 0:
        offset = torch.randn(x_1.shape[0], x_1.shape[1], 1, 1, device=device, dtype=dtype)
        x_0 = x_0 + offset_noise * offset
        
    x_t = (1.0 - t.view(-1,1,1,1)) * x_0 + t.view(-1,1,1,1) * x_1
    target = x_1 - x_0
    
    return x_t, t, x_1, target, text, text_mask

def get_base_loss(v_pred, target, loss_type):
    if loss_type == "mse":
        return F.mse_loss(v_pred, target)
    elif loss_type == "l1":
        return F.l1_loss(v_pred, target)
    elif loss_type == "huber":
        return F.huber_loss(v_pred, target, delta=0.1)
    else:
        return F.mse_loss(v_pred, target)

def get_fourier_amplitude_loss(x_hat_1, x_1, t, fal_lambda=0.05):
    time_weight = t.view(-1, 1, 1, 1) ** 2
    
    x_hat_fft = torch.fft.fft2(x_hat_1.float(), dim=(-2, -1), norm='ortho')
    x_true_fft = torch.fft.fft2(x_1.float(), dim=(-2, -1), norm='ortho')
    
    loss_fal_raw = F.mse_loss(torch.abs(x_hat_fft), torch.abs(x_true_fft), reduction='none')
    loss_fal = (loss_fal_raw * time_weight).mean()
    
    return fal_lambda * loss_fal

def get_fourier_correlation_loss(x_hat_1, x_1, t, fcl_lambda=0.05):
    time_weight = t.view(-1, 1, 1, 1) ** 2
    
    x_hat_flat = x_hat_1.view(x_hat_1.size(0), -1).float()
    x_1_flat = x_1.view(x_1.size(0), -1).float()
    
    correlation = F.cosine_similarity(x_hat_flat, x_1_flat, dim=1)
    loss_fcl_raw = 1.0 - correlation.clamp(-1.0, 1.0)
    
    loss_fcl = (loss_fcl_raw.view(-1, 1, 1, 1) * time_weight).mean()
    
    return fcl_lambda * loss_fcl

def get_self_eval_loss(x_hat_1, x_1, t, s, ema_model, text, text_mask, self_eval_lambda, cfg_val=1.5):
    with torch.no_grad():
        noise_s = torch.randn_like(x_hat_1)
        x_hat_s = (1.0 - s.view(-1, 1, 1, 1)) * noise_s + s.view(-1, 1, 1, 1) * x_hat_1
        
        teacher_net = ema_model.module if hasattr(ema_model, 'module') else ema_model

        text_uncond = torch.zeros_like(text)
        combined_text = torch.cat([text_uncond, text], dim=0)
        combined_mask = torch.cat([text_mask, text_mask], dim=0)
        
        x_self_teacher = cfg_guided_position(teacher_net, x_hat_s, s, combined_text, cfg_val, combined_mask)
        
        x_self = x_hat_1 + (x_self_teacher - x_hat_s)
        
        lambd_weight = (s / (1.0 - s + 1e-4)) - (t / (1.0 - t + 1e-4))
        lambd_weight = lambd_weight.view(-1, 1, 1, 1).clamp(0, 10)
        
        target_raw = x_1 + lambd_weight * x_self
        
        norm_clean = torch.linalg.vector_norm(x_1, dim=(1, 2, 3), keepdim=True)
        norm_target = torch.linalg.vector_norm(target_raw, dim=(1, 2, 3), keepdim=True)
        norm_factor = norm_clean / (norm_target + 1e-6)
        
        x_renorm = target_raw * norm_factor
    
    loss_self = F.mse_loss(x_hat_1, x_renorm.detach())
    
    return loss_self * self_eval_lambda

def calculate_total_loss(model, ema_model, x_t, t, x_1, target, text, text_mask, epoch, epochs, use_self_eval, 
                         start_self_eval_at, self_eval_lambda, fal_lambda, fcl_lambda, loss_type):
    v_pred = model(x_t, t, text, text_mask)
    
    loss_real = get_base_loss(v_pred, target, loss_type)
        
    loss_fal = 0.0
    loss_fcl = 0.0
    x_hat_1 = None 
    
    need_x_hat = (fal_lambda > 0) or (fcl_lambda > 0) or (use_self_eval and epoch > (epochs * start_self_eval_at))

    if need_x_hat:
        x_hat_1 = predict_x1_from_velocity(x_t, t, v_pred)
        
        if fal_lambda > 0:
            loss_fal = get_fourier_amplitude_loss(x_hat_1, x_1, t, fal_lambda=fal_lambda)
            
        if fcl_lambda > 0:
            loss_fcl = get_fourier_correlation_loss(x_hat_1, x_1, t, fcl_lambda=fcl_lambda)
    
    loss = loss_real + loss_fal + loss_fcl
    
    if use_self_eval and epoch > (epochs * start_self_eval_at):
        s = t + torch.rand_like(t) * (1.0 - t)
        loss_self = get_self_eval_loss(x_hat_1, x_1, t, s, ema_model, text, text_mask, self_eval_lambda, cfg_val=1.5)                    
        loss = loss + loss_self                    
    
    #print(f"Loss Real: {loss_real}| Loss Fal {loss_fal}| Loss Fcl {loss_fcl}")
    
    return loss