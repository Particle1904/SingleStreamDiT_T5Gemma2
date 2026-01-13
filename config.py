import os
import torch

class Config:
    # ============================================================
    #
    #                          IMPORTANT
    #
    # - VAE scaling + latent normalization MUST remain consistent across
    #   preprocess.py, train.py, and inference.
    # - Changing model dimensions invalidates checkpoints.
    # ============================================================
    # REGION: PROJECT & PATHS
    # General experiment metadata and filesystem layout
    # ============================================================
    project_name = "flowers"
    output_dir = "./output"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    samples_dir = os.path.join(output_dir, "samples")
    log_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(log_dir, f"{project_name}_log.csv")    
    dataset_dir = "./dataset"
    cache_dir = "./cached_data"    
    # Used by sanity_check / cache inspection utilities
    target_file = os.path.join(cache_dir, "39.pt")        
    # Resume training from a full checkpoint (model + optimizer + EMA)
    # Set to None for a fresh run
    resume_from = None
    # Reset optimizer when resume training
    reset_optmizer = True
        
    # ============================================================
    # REGION: MODEL ARCHITECTURE
    # Core DiT / transformer structure (checkpoint-breaking changes)
    # ============================================================
    # Text encoder output size
    # 640  -> T5Gemma2-270M-270M
    # 1152 -> T5Gemma2-1B-1B
    # 2560 -> T5Gemma2-4B-4B
    text_embed_dim = 1152
    # FLUX VAE latent channels (FLUX.1 uses 16)
    in_channels = 16    

    # DiT backbone
    hidden_size = 384
    num_heads = 6
    depth = 20
    # Separate refinement stages
    refiner_depth = 2
    # Max token length for text conditioning
    max_token_length = 128
    # Patch size in latent space (latent pixels per token)
    patch_size = 2
    # Rotary embedding base
    rope_base = 10_000
    
    # ============================================================
    # REGION: EXTERNAL MODELS
    # HuggingFace / Diffusers model identifiers
    # ============================================================
    vae_id = "diffusers/FLUX.1-vae"
    text_model_id = "google/t5gemma-2-1b-1b"
    
    # ============================================================
    # REGION: PREPROCESSING & LATENT CONVENTIONS
    # MUST MATCH across preprocess / train / inference
    # ============================================================
    # Target training resolution (area-preserving bucketing)
    target_resolution = 448
    # Buckets aligned to multiples of this value
    bucket_alignment = 32
    # FLUX VAE scaling factor (Diffusers default for FLUX)
    # Latents are MULTIPLIED by this during encode
    vae_scaling_factor = 0.3611
    # Spatial downsample factor of the VAE
    # Used to compute latent H/W from image H/W
    vae_downsample_factor= 8
    # Dataset-wide latent normalization (computed post-preprocess)
    # normalize: (x - mean) / std
    # After testing it extensively, just using 0.0 and 1.0 results in better reconstructed images
    # By just using calculate_vae_statistics.py and changing the values below, the reconstructed images
    # get a very weak blue tint effect and tiling pattern.
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
    epochs = 1500
    # Effective batch size per optimizer step
    batch_size = 12
    accum_steps = 1
    # Loss for velocity prediction
    # Options: "mse", "l1", "huber"
    loss_type = "mse"
    
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
    # REGION: FLOW MATCHING & SAMPLING
    # Time parameterization and numerical integration
    # ============================================================
    shift_val = 1.0        
    
    # ============================================================
    # REGION: SELF-Evaluation (EXPERIMENTAL)
    # Teacher–student consistency regularization
    # ============================================================
    # Enable Self-Evaluation (recommended OFF for initial training)
    use_self_eval = False
    # Fraction of total epochs before Self-Evaluation activates
    start_self_eval_at = 0.75
    # Strength of self-evaluation loss
    self_eval_lambda = 0.3
    
    # ============================================================
    # REGION: OPTIMIZATION & PRECISION
    # Runtime and numerical behavior
    # ============================================================
    dtype = torch.bfloat16
    gradient_checkpointing = False
    # Exponential Moving Average for inference stability
    use_ema = True
    ema_decay = 0.999
    
    # ============================================================
    # REGION: SYSTEM & DATALOADING
    # ============================================================
    device = "cuda"
    # Cache entire dataset in RAM (recommended for <= ~20k images)
    load_entire_dataset = True
    num_workers = 4 if os.name != 'nt' else 0
    
    # ============================================================
    # REGION: LOGGING & VALIDATION
    # ============================================================
    save_every = 100
    validate_every = 25
    # Validation sampling parameters
    validate_cfg = 1.25
    validate_steps = 30 
    validate_sampler = "euler"
    
    # ============================================================
    # REGION: INFERENCE DEFAULTS
    # Used by inference scripts / notebooks
    # ============================================================
    inference_steps = 50
    guidance_scale = 3.5
    # "euler" or "rk4"
    sampler = "rk4"       