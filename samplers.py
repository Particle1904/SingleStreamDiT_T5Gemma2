import torch

def get_1d_shifted_time(t, shift_val):
    if shift_val == 1.0:
        return t
    return (t * shift_val) / (1 + (shift_val - 1) * t)

def cfg_velocity(model, x, t, text_embeds, cfg, text_mask=None):
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    v_out = model(x_in, t_in, text_embeds, text_mask=text_mask) 
    v_uncond, v_cond = v_out.chunk(2, dim=0)
    return v_uncond + cfg * (v_cond - v_uncond)

def euler_step(model, x, t, dt, text_embeds, cfg, text_mask=None):
    v = cfg_velocity(model, x, t, text_embeds, cfg, text_mask=text_mask)
    return x + v * dt

def rk4_step(model, x, t, dt, text_embeds, cfg, t_mid, t_end, text_mask=None):
    k1 = cfg_velocity(model, x, t, text_embeds, cfg, text_mask)
    k2 = cfg_velocity(model, x + 0.5 * dt * k1, t_mid, text_embeds, cfg, text_mask)
    k3 = cfg_velocity(model, x + 0.5 * dt * k2, t_mid, text_embeds, cfg, text_mask)
    k4 = cfg_velocity(model, x + dt * k3, t_end, text_embeds, cfg, text_mask)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def predict_x1_from_velocity(x, t, v):
    return x + (1.0 - t.view(-1, 1, 1, 1)) * v

# Used for SELF-EVAL
def cfg_guided_position(model, x, t, text_embeds, cfg=1.0, text_mask=None):
    v = cfg_velocity(model, x, t, text_embeds, cfg, text_mask=text_mask)
    return predict_x1_from_velocity(x, t, v)

def run_sampling_pipeline(model, initial_noise, steps, combined_text_embeds, cfg, sampler_type, shift_val, text_mask=None):
    x = initial_noise.clone()
    dt = 1.0 / steps

    combined_text = combined_text_embeds 
    
    for i in range(steps):
        t_linear = torch.tensor([i / steps], device=x.device, dtype=x.dtype)
        t = get_1d_shifted_time(t_linear, shift_val)

        if sampler_type == "euler":
            x = euler_step(model, x, t, dt, combined_text, cfg, text_mask)                
        elif sampler_type == "rk4":
            t_mid_linear = torch.tensor([(i + 0.5) / steps], device=x.device, dtype=x.dtype)
            t_mid = get_1d_shifted_time(t_mid_linear, shift_val)
            
            t_end_linear = torch.tensor([(i + 1.0) / steps], device=x.device, dtype=x.dtype)
            t_end = get_1d_shifted_time(t_end_linear, shift_val)
            x = rk4_step(model, x, t, dt, combined_text, cfg, t_mid, t_end, text_mask)
        else:
            raise ValueError(f"Unknown sampler: {sampler_type}")
            
    return x