import torch
from config import Config

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