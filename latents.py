import torch
from config import Config
from PIL import Image

def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
    return (latents - Config.dataset_mean) / Config.dataset_std

def denormalize_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents * Config.dataset_std + Config.dataset_mean

def to_vae_space(latents: torch.Tensor) -> torch.Tensor:
    return latents

def from_vae_space(latents: torch.Tensor) -> torch.Tensor:
    return latents

def prepare_latents_for_decode(latents: torch.Tensor, clamp=False, print_debug=False) -> torch.Tensor:
    latents = denormalize_latents(latents)
    latents = to_vae_space(latents)

    if clamp:
        latents = torch.clamp(latents, -4.0, 4.0)
        if print_debug:
            print(f"Latents Min: {latents.min().item():.2f}, Max: {latents.max().item():.2f}")
       
    if print_debug:
        print(f"Latents after torch.clamp Min: {latents.min().item():.2f}, Max: {latents.max().item():.2f}")
    return latents

def decode_latents_to_image(vae_model, latents: torch.Tensor, device) -> Image.Image:
    latents = prepare_latents_for_decode(latents)
    with torch.no_grad():
        device_type = device.type if isinstance(device, torch.device) else device
        with torch.autocast(device_type, enabled=False):
            decoded = vae_model.decode(latents.float())
            image_tensor = decoded.sample if hasattr(decoded, "sample") else decoded[0]
            
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    image_tensor = (image_tensor * 255).round().astype("uint8")
    return Image.fromarray(image_tensor[0])

@torch.no_grad()
def get_combined_text_embeds(prompt: str, neg_prompt: str, cfg: float, text_encoder):
    def encode_single(p: str):
        embeds, mask = text_encoder.encode(p)
        return embeds.squeeze(0), mask.squeeze(0)
    
    cond_embeds, cond_mask = encode_single(prompt)
    
    if neg_prompt and cfg > 1.0:
        uncond_embeds, uncond_mask = encode_single(neg_prompt)
    else:
        uncond_embeds = torch.zeros_like(cond_embeds)
        uncond_mask = torch.ones_like(cond_mask)
    
    combined_text = torch.cat([uncond_embeds.unsqueeze(0), cond_embeds.unsqueeze(0)], dim=0)
    combined_mask = torch.cat([uncond_mask.unsqueeze(0), cond_mask.unsqueeze(0)], dim=0)
    
    return combined_text, combined_mask

def load_repa_target(data: dict, dtype: torch.dtype) -> torch.Tensor:
    repa_target_quant = data.get("repa_target", None)
    if repa_target_quant is not None:
        if repa_target_quant.dtype == torch.int8:
            repa_scale = data.get("repa_scale", torch.tensor(1.0))
            return repa_target_quant.to(dtype) * repa_scale.to(dtype)
        return repa_target_quant.to(dtype)
    return torch.zeros(1).to(dtype)