import torch

def get_1d_shifted_time(t, shift_val):
    if shift_val == 1.0:
        return t
    return (t * shift_val) / (1 + (shift_val - 1) * t)

def cfg_velocity(model, x, t, text_embeds, cfg):
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    v_out = model(x_in, t_in, text_embeds)
    v_uncond, v_cond = v_out.chunk(2, dim=0)
    return v_uncond + cfg * (v_cond - v_uncond)

def euler_step(model, x, t, dt, text_embeds, cfg):
    v = cfg_velocity(model, x, t, text_embeds, cfg)
    return x + v * dt

def rk4_step(model, x, t, dt, text_embeds, cfg, t_mid):
    k1 = cfg_velocity(model, x, t, text_embeds, cfg)
    k2 = cfg_velocity(model, x + 0.5 * dt * k1, t_mid, text_embeds, cfg)
    k3 = cfg_velocity(model, x + 0.5 * dt * k2, t_mid, text_embeds, cfg)
    k4 = cfg_velocity(model, x + dt * k3, t_mid, text_embeds, cfg)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def euler_to_1(x, t, v):
    return x + (1.0 - t.view(-1, 1, 1, 1)) * v

# Used for SELF-EVAL
def cfg_guided_position(model, x, t, text_embeds, cfg=1.0):
    v = cfg_velocity(model, x, t, text_embeds, cfg)
    return euler_to_1(x, t, v)