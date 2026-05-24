import torch
import torch.nn.functional as F
from samplers import get_1d_shifted_time  

def prepare_batch_and_targets(batch, device, dtype, shift_val):
    x_1 = batch["latents"].to(device, dtype=dtype)
    text = batch["text_embeds"].to(device, dtype=dtype)
    text_mask = batch["text_mask"].to(device)
    u = torch.rand(x_1.shape[0], device=device, dtype=dtype)
    t = get_1d_shifted_time(u, shift_val)
    x_0 = torch.randn_like(x_1)
    x_t = (1.0 - t.view(-1,1,1,1)) * x_0 + t.view(-1,1,1,1) * x_1
    target = x_1 - x_0
    return x_t, t, x_1, target, text, text_mask

def get_base_loss(v_pred, target, loss_type):
    if loss_type == "mse":
        return F.mse_loss(v_pred, target, reduction='none').mean(dim=(1, 2, 3))
    elif loss_type == "l1":
        return F.l1_loss(v_pred, target, reduction='none').mean(dim=(1, 2, 3))
    elif loss_type == "huber":
        return F.huber_loss(v_pred, target, delta=0.1, reduction='none').mean(dim=(1, 2, 3))
    else:
        return F.mse_loss(v_pred, target, reduction='none').mean(dim=(1, 2, 3))

def calculate_total_loss(model, x_t, t, target, text, text_mask, loss_type, repa_target=None, repa_lambda = 0.5):
    if repa_target is not None:
        v_pred, repa_pred = model(x_t, t, text, text_mask, return_repa=True)
        base_loss = get_base_loss(v_pred, target, loss_type)
        
        cos_sim = F.cosine_similarity(repa_pred, repa_target, dim=-1)
        repa_loss = (1.0 - cos_sim).mean(dim=1)
        
        total_loss = base_loss + repa_lambda * repa_loss
        
        return total_loss, base_loss, repa_loss
    else:
        v_pred = model(x_t, t, text, text_mask)
        return get_base_loss(v_pred, target, loss_type)