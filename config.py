import os
import torch
import argparse
import sys

class Config:
    # ============================================================
    #
    #                          IMPORTANT
    #
    # - VAE scaling + latent normalization MUST remain consistent 
    #   across preprocess.py, train.py, and inference.
    # - Changing model dimensions invalidates checkpoints.
    # ============================================================
    # REGION: PROJECT & PATHS
    # General experiment metadata and filesystem layout
    # ============================================================
    seed = 42
    project_name = "flowers"
    
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    if is_kaggle:
        output_dir = "/kaggle/working/output"
        cache_dir = "/kaggle/input/oxfordflowers/cached_data"
        print("Kaggle kernel detected.")
    else:
        output_dir = "./output"
        cache_dir = "./cached_data"  
        print("Not in Kaggle kernel.")
    dataset_dir = "./dataset"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    samples_dir = os.path.join(output_dir, "samples")
    log_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(log_dir, f"{project_name}_log.csv")    
    
    # Used by sanity_check / cache inspection utilities
    target_filename = "39.pt"
    target_file = os.path.join(cache_dir, target_filename)        
    # Resume training from a full checkpoint (model + optimizer + EMA)
    # Set to None for a fresh run or "latest" for HF model
    resume_from = None
    # Reset optimizer when resume training
    reset_optimizer = False
        
    # ============================================================
    # REGION: MODEL ARCHITECTURE
    # Core DiT / transformer structure (checkpoint-breaking changes)
    # ============================================================
    # Text encoder output size
    # 640  -> T5Gemma2-270M-270M
    # 1152 -> T5Gemma2-1B-1B
    # 2560 -> T5Gemma2-4B-4B
    text_embed_dim = 1152

    # DiT backbone
    hidden_size = 768
    num_heads = 12
    depth = 16
    # Separate refinement stages
    refiner_depth = 4
    # Max token length for text conditioning
    max_token_length = 256
    # Patch size in latent space (latent pixels per token)
    patch_size = 2
    # Rotary embedding base
    rope_base = 10_000
    use_xsa = False
    
    # ============================================================
    # REGION: EXTERNAL MODELS
    # HuggingFace / Diffusers model identifiers
    # ============================================================
    # "diffusers/FLUX.1-vae"
    vae_id = "kaiyuyue/FLUX.2-dev-vae"
    # FLUX VAE latent channels (FLUX.1 uses 16, FLUX.2 uses 32, SD VAE uses 4)
    in_channels = 32
    preprocess_batch_size = 4
    text_model_id = "google/t5gemma-2-1b-1b"
    
    # ============================================================
    # REGION: PREPROCESSING & LATENT CONVENTIONS
    # MUST MATCH across preprocess / train / inference
    # ============================================================
    # Target training resolution (area-preserving bucketing)
    target_resolution = 512
    # Buckets aligned to multiples of this value
    bucket_alignment = 32
    # FLUX VAE scaling factor (Diffusers default for FLUX)
    # Latents are MULTIPLIED by this during encode
    # NOTE: FLUX.2-dev-vae (AutoencoderKLFlux2) normalization is handled
    # internally by the VAE. to_vae_space / from_vae_space are identity.
    vae_scaling_factor = 0.3611
    # Spatial downsample factor of the VAE
    # Used to compute latent H/W from image H/W
    vae_downsample_factor = 8
    # Dataset-wide latent normalization (computed post-preprocess)
    # normalize: (x - mean) / std
    # After testing it extensively, just using 0.0 and 1.0 results 
    # in better reconstructed image. By just using calculate_vae_statistics.py 
    # and changing the values below, the reconstructed images get a very weak 
    # blue tint effect and tiling pattern.
    dataset_mean = 0.0
    dataset_std = 1.0

    # ============================================================
    # REGION: TRAINING HYPERPARAMETERS
    # Optimization and regularization behavior
    # ============================================================
    # Base learning rate (AdamW / 8-bit Adam)
    # 1e-4 or 2e-4 for fresh/aggressive and 4e-5 or 5e-5 for fine-tuning
    learning_rate = 1e-4
    # Total number of epochs (from scratch or resumed)
    epochs = 1400
    # Effective batch size per optimizer step
    batch_size = 32
    accum_steps = 2
    # Loss for velocity prediction
    # Options: "mse", "l1", "huber". "edm"
    loss_type = "edm"
    
    # Transformer regularization
    model_dropout = 0.05
    weight_decay = 0.05
    optimizer_warmup = 0.05
    offset_noise = 0.05
    # Drop text conditioning during training (CFG support)    
    text_dropout = 0.15
    # Random horizontal flip in latent space
    flip_aug = False 
    
    # ============================================================
    # REGION: OPTIMIZATION & PRECISION
    # Runtime and numerical behavior
    # ============================================================
    dtype = torch.bfloat16
    gradient_checkpointing = True
    # Exponential Moving Average for inference stability
    use_ema = True
    ema_decay = 0.999
    
    # ============================================================
    # REGION: FLOW MATCHING & SAMPLING
    # Time parameterization and numerical integration
    # ============================================================
    # 3.0 for FLUX1 VAE, 4.63-6.93 for FLUX2 VAE
    shift_val = 4.69    
    
    # ============================================================
    # REGION: SELF-Evaluation (EXPERIMENTAL)
    # Teacher–student consistency regularization
    # ============================================================
    # Enable Self-Evaluation (recommended OFF for initial training)
    use_self_eval = False
    # Fraction of total epochs before Self-Evaluation activates
    start_self_eval_at = 0.85
    # Strength of self-evaluation loss
    self_eval_lambda = 0.3
    
    # ============================================================
    # REGION: FOURIER LOSSES
    # Fourier Amplitude Loss lambda
    fal_lambda = 0.0
    # Fourier Correlation Loss lambda
    fcl_lambda = 0.0
                
    # ============================================================
    # REGION: SYSTEM & DATALOADING
    # ============================================================
    accelerator = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cache entire dataset in RAM (recommended for <= ~20k images)
    load_entire_dataset = True
    num_workers = 2 if os.name != 'nt' else 0
    
    # ============================================================
    # REGION: LOGGING & VALIDATION
    # ============================================================
    run_validation_loss = False 
    save_every = 100
    validate_every = 25
    # Validation sampling parameters
    validate_cfg = 5.0
    validate_steps = 30 
    validate_sampler = "euler"
    validate_scheduler = "uniform"
    
    # ============================================================
    # REGION: INFERENCE DEFAULTS
    # Used by inference scripts / notebooks
    # ============================================================
    inference_steps = 50
    guidance_scale = 3.5
    # Options: "euler" or "rk4" or "dpmpp"
    sampler = "rk4"
    # Options: "uniform" or "karras" or "beta"
    scheduler = "karras"
    
    # ============================================================
    # REGION: HUGGINGFACE INTEGRATION
    # ============================================================
    # How many checkpoints to keep in HF
    keep_last = 4
    push_to_hub = False
    hf_repo_id = "Crowlley/SingleStreamDiT" 
    hf_token = os.environ.get("HF_TOKEN")
    
    compile_model = False
    
def parse_config_args():
    parser = argparse.ArgumentParser(description="Override Config parameters")
    for key, value in Config.__dict__.items():
        if not key.startswith("__") and not callable(value):
            if isinstance(value, bool):
                parser.add_argument(f"--{key}", type=lambda x: (str(x).lower() == 'true'), default=value)
            else:
                parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    
    args, unknown = parser.parse_known_args()
    
    for key, value in vars(args).items():
        setattr(Config, key, value)