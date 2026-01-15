import torch
from config import Config
from PIL import Image

def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
    return (latents - Config.dataset_mean) / Config.dataset_std

def denormalize_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents * Config.dataset_std + Config.dataset_mean

def to_vae_space(latents: torch.Tensor) -> torch.Tensor:
    return latents / Config.vae_scaling_factor

def from_vae_space(latents: torch.Tensor) -> torch.Tensor:
    return latents * Config.vae_scaling_factor

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

def decode_latents_to_image(vae_model, latents: torch.Tensor, device: str) -> Image.Image:
    latents = prepare_latents_for_decode(latents)

    with torch.no_grad():
        with torch.autocast(device, enabled=False): 
            image_tensor = vae_model.decode(latents.float()).sample

    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    image_tensor = (image_tensor * 255).round().astype("uint8")
    
    return Image.fromarray(image_tensor[0])

def get_combined_text_embeds(prompt: str, neg_prompt: str, cfg: float, tokenizer, text_encoder, 
                             max_token_length: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    
    # Positive Embedding
    inputs_cond = tokenizer(prompt, max_length=max_token_length, padding="max_length", truncation=True, 
                            return_tensors="pt").to(device)
    out_cond = text_encoder(input_ids=inputs_cond.input_ids, attention_mask=inputs_cond.attention_mask)
    cond_embeds = out_cond.last_hidden_state if hasattr(out_cond, "last_hidden_state") else out_cond[0]
    
    # Negative Embedding, fall back to empty if no prompt
    if neg_prompt and cfg > 1.0:
        inputs_uncond = tokenizer(neg_prompt, max_length=max_token_length, padding="max_length", truncation=True, 
                                  return_tensors="pt").to(device)
        out_uncond = text_encoder(input_ids=inputs_uncond.input_ids, attention_mask=inputs_uncond.attention_mask)
        uncond_embeds = out_uncond.last_hidden_state if hasattr(out_uncond, "last_hidden_state") else out_uncond[0]
    else:
        uncond_embeds = torch.zeros_like(cond_embeds)
    
    combined_text = torch.cat([uncond_embeds, cond_embeds], dim=0).to(dtype=dtype)
    
    return combined_text