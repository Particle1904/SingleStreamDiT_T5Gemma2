import torch
import torch.nn.functional as F
from samplers import cfg_guided_position, predict_x1_from_velocity, get_1d_shifted_time  

def prepare_batch_and_targets(batch, device, dtype, shift_val, offset_noise):
    x_1 = batch["latents"].to(device, dtype=dtype)
    text = batch["text_embeds"].to(device, dtype=dtype)
    
    u = torch.rand(x_1.shape[0], device=device, dtype=dtype)
    t = get_1d_shifted_time(u, shift_val)            
    x_0 = torch.randn_like(x_1)
    x_0 = x_0 + offset_noise * torch.randn(x_1.shape[0], x_1.shape[1], 1, 1, device=device, dtype=dtype)            
    
    x_t = (1.0 - t.view(-1,1,1,1)) * x_0 + t.view(-1,1,1,1) * x_1
    # v_target = x_1 - x_0
    target = x_1 - x_0

    return x_t, t, x_1, target, text

def get_base_loss(v_pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(v_pred, target)
    elif loss_type == "l1":
        return F.l1_loss(v_pred, target)
    elif loss_type == "huber":
        return F.huber_loss(v_pred, target, delta=0.1)
    else:
        # Just return mse_loss
        return F.mse_loss(v_pred, target)

def get_fourier_amplitude_loss(x_hat_1: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, 
                               fal_lambda: float = 0.05) -> torch.Tensor:
    fal_curriculum_mask = (t > 0.5).view(-1, 1, 1, 1)

    time_weight = t.view(-1, 1, 1, 1) ** 2 

    x_hat_fft = torch.fft.rfft2(x_hat_1.float(), dim=(-2, -1), norm='ortho')
    x_true_fft = torch.fft.rfft2(x_1.float(), dim=(-2, -1), norm='ortho')
    
    loss_fal_raw = F.mse_loss(torch.abs(x_hat_fft), torch.abs(x_true_fft), reduction='none')    
    loss_fal = (loss_fal_raw * fal_curriculum_mask * time_weight).mean()
    
    return fal_lambda * loss_fal

def get_fourier_correlation_loss(x_hat_1: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, 
                                 fcl_lambda: float = 0.05) -> torch.Tensor:
    fcl_curriculum_mask = (t > 0.3).view(-1, 1, 1, 1)
    
    x_hat_1_float = x_hat_1.float()
    x_1_float = x_1.float()

    F = torch.fft.rfft2(x_1_float, dim=(-2, -1), norm='ortho')
    F_hat = torch.fft.rfft2(x_hat_1_float, dim=(-2, -1), norm='ortho')

    numerator_complex = F * torch.conj(F_hat) 
    numerator_real_sum = torch.sum(numerator_complex.real, dim=(-3, -2, -1))

    F_abs_sq_sum = torch.sum(torch.abs(F)**2, dim=(-3, -2, -1))
    F_hat_abs_sq_sum = torch.sum(torch.abs(F_hat)**2, dim=(-3, -2, -1))
    denominator = torch.sqrt(F_abs_sq_sum * F_hat_abs_sq_sum)

    correlation = numerator_real_sum / (denominator + 1e-6)
    correlation = torch.clamp(correlation, -1.0, 1.0) 
    
    loss_fcl_raw = 1.0 - correlation
    loss_fcl = (loss_fcl_raw.view(-1, 1, 1, 1) * fcl_curriculum_mask).mean()

    return fcl_lambda * loss_fcl

def get_self_eval_loss(x_hat_1: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, s: torch.Tensor, 
                       ema_model: torch.nn.Module, text: torch.Tensor, self_eval_lambda: float, 
                       cfg_val: float = 1.5) -> torch.Tensor:
    # Note: This function is expected to be called inside torch.no_grad() for the teacher part
    noise_s = torch.randn_like(x_hat_1)
    x_hat_s = (1.0 - s.view(-1, 1, 1, 1)) * noise_s + s.view(-1, 1, 1, 1) * x_hat_1
    
    teacher_net = ema_model.module if hasattr(ema_model, 'module') else ema_model
    
    text_uncond = torch.zeros_like(text)
    combined_text = torch.cat([text_uncond, text], dim=0)
    
    with torch.no_grad():
        x_self = cfg_guided_position(model=teacher_net, x=x_hat_s, t=s, text_embeds=combined_text, cfg=cfg_val)
    x_self = x_hat_1 + (x_self - x_hat_s)
    
    lambd_weight = (t / (1.0 - t + 1e-4)) - (s / (1.0 - s + 1e-4))
    lambd_weight = lambd_weight.view(-1, 1, 1, 1).clamp(0, 10)
    
    target_raw = x_1 + lambd_weight * x_self
    
    norm_clean = torch.linalg.vector_norm(x_1, dim=(1, 2, 3), keepdim=True)
    norm_target = torch.linalg.vector_norm(target_raw, dim=(1, 2, 3), keepdim=True)
    norm_factor = norm_clean / (norm_target + 1e-6)
    x_renorm = target_raw * norm_factor
    
    loss_self = F.mse_loss(x_hat_1, x_renorm)
    
    return loss_self * self_eval_lambda

def calculate_total_loss(model, ema_model, x_t, t, x_1, target, text, epoch, epochs, use_self_eval, 
                         start_self_eval_at, self_eval_lambda, fal_lambda, fcl_lambda, loss_type, accum_steps):
    # Model forward pass
    v_pred = model(x_t, t, text)
    
    # Base Velocity Loss
    loss_real = get_base_loss(v_pred, target, loss_type)

    # Initialize FFT Vars
    loss_fal = 0.0
    loss_fcl = 0.0
    x_hat_1 = None 
    
    # Check if we need x_hat_1 for Fourier OR Self-Eval
    need_x_hat = (fal_lambda > 0) or (fcl_lambda > 0) or (use_self_eval and epoch > (epochs * start_self_eval_at))

    if need_x_hat:
        # Predict the final image from the current velocity
        x_hat_1 = predict_x1_from_velocity(x_t, t, v_pred)
        
        # Calculate Fourier Losses if enabled
        if fal_lambda > 0:
            loss_fal = get_fourier_amplitude_loss(x_hat_1, x_1, t, fal_lambda=fal_lambda)
            
        if fcl_lambda > 0:
            loss_fcl = get_fourier_correlation_loss(x_hat_1, x_1, t, fcl_lambda=fcl_lambda)
    
    # Initialize total loss
    loss = loss_real + loss_fal + loss_fcl 
    
    # Self-Evaluation Loss
    if use_self_eval and epoch > (epochs * start_self_eval_at):
        s = t + torch.rand_like(t) * (1.0 - t)
        loss_self = get_self_eval_loss(x_hat_1=x_hat_1, x_1=x_1, t=t, s=s, 
                                       ema_model=ema_model, text=text, 
                                       self_eval_lambda=self_eval_lambda, cfg_val=1.5)                    
        loss = loss + loss_self                    
    
    return loss / accum_steps