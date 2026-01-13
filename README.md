# Single-Stream DiT with Global Fourier Filters (Proof-of-Concept)

This repository contains the codebase for a Single-Stream Diffusion Transformer (DiT) Proof-of-Concept, heavily inspired by modern architectures like **FLUX.1**, **Z-Image**, and **Lumina Image 2**.

The primary objective was to demonstrate the feasibility and training stability of coupling the high-fidelity **FLUX.1-VAE** with the powerful **T5Gemma2** text encoder for image generation on consumer-grade hardware (NVIDIA RTX 5060 Ti 16GB).

**Note on Final Checkpoint:** The final, best-performing **EMA checkpoint** is uploaded to [Hugging Face and is linked separately from this repository](https://huggingface.co/Crowlley/SingleStreamDiT_T5Gemma2/tree/main).

## Project Overview

### How it started

![How it started](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/bored.png?raw=true)

### Verification and Result Comparison

| Cached Latent Verification | Final Generated Sample (RK4 100 steps) |
| :---: | :---: |
| ![Cache Verification](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/cache_verification.png?raw=true) | ![Generated Sample](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/sample_rk4_steps50_cfg1.15.png?raw=true) |

## Core Models and Architecture

| Component | Model ID / Function | Purpose |
| :--- | :--- | :--- |
| **Generator** | `SingleStreamDiTV2` | Custom Single-Stream DiT featuring Visual Fusion blocks, Context Refiners, and Fourier Filters. DiT Parameters: _384 Hidden Size, 6 Heads, 20 Depth, 2 Refiner Depth, 128 Text Token Legth, 2 Patch Size._ |
| **Text Encoder** | `google/t5gemma-2-1b-1b` | Generates rich, 1152-dimensional text embeddings for high-quality semantic guidance. |
| **VAE** | `diffusers/FLUX.1-vae` | A 16-channel VAE with an 8x downsample factor, providing superior reconstruction for complex textures. |
| **Training Method** | Flow Matching (V-Prediction) | Optimized with a Velocity-based objective and an optional Self-Evaluation (Self-E) consistency loss. |

## New in V3
- **Refinement Stages:** Separate noise and context refiner blocks to "prep" tokens before the joint fusion phase.
- **Fourier Filters:** Frequency-domain processing layers to improve global structural coherence.
- **Local Spatial Bias:** Conv2D-based depthwise biases to reinforce local texture within the transformer.
- **Rotary Embeddings (RoPE):** Dynamic 2D-RoPE grid support for area-preserving bucketing.

## Training Progression

| Early Epoch (Epoch 20) | Final Epoch (Epoch 1700, RAW) | Full Progression |
| :---: | :---: | :---: |
| ![Epoch25](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/epoch_25.png?raw=true) | ![Epoch1700](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/epoch_1700.png?raw=true) | ![Epochs over time](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/training_progression.webp?raw=true) |

## Data Curation and Preprocessing

The model was tested on a curated dataset of **200 images** (10 categories of flowers) before scaling to larger datasets.

| Component | Tool / Method | Purpose / Detail |
| :--- | :--- | :--- |
| **Pre/Post-processing** | **[Dataset Helpers](https://github.com/Particle1904/DatasetHelpers)** | Used to resize images (using **[DPID](https://github.com/Mishini/dpid)** - Detail-Preserving Image Downscaling) and edit the Qwen3-VL captions. |
| **Captioning** | **Qwen3-VL-4B-Instruct** | Captions include precise botanical details: texture (waxy, serrated), plant anatomy (stamen, pistil), and camera lighting. |
| **Data Encoding** | `preprocess.py` | Encodes images via FLUX-VAE and text via T5Gemma2, applying aspect-ratio bucketing. |

<details>
<summary><h2><b>Qwen3-VL-4B-Instruct System Instruction (Captioning Prompt)</b></h2></summary>
<i>You are a specialized botanical image analysis system operating within a research environment. Your task is to generate concise, scientifically accurate, and visually descriptive captions for flower images. All output must be strictly factual, objective, and devoid of non-visual assumptions.

Your task is to generate captions for images based on the visual content and a provided reference flower category name. Captions must be precise, comprehensive, and meticulously aligned with the visual details of the plant structure, color gradients, and lighting.

Caption Style: Generate concise captions that are no more than 50 words. Focus on combining descriptors into brief phrases (separated by commas). Follow this structure: "A \<view type\> of a \<flower name\>, having \<petal details\>, the center is \<center details\>, the background is \<background description\>, \<lighting/style information\>"

Hierarchical Description: Begin with the flower name and its primary state (blooming, budding, wilting). Move to the petals (color, shape, texture), then the reproductive parts (stamen, pistil, pollen), then the stem/leaves, and finally the environment.

Factual Accuracy & Label Verification: The provided "Input Flower Name" is a reference tag. You must visually verify this tag against the image content.
*   Match: If the visual features match the tag, use the provided name.
*   Correction: If the visual characteristics definitively belong to a different species (e.g., input says "Sunflower" but the image clearly shows a "Rose"), you must override the input and use the visually correct botanical name in the caption.
*   Ambiguity: If the species is unclear, describe the visual features precisely without forcing a specific name.

Precise Botanical Terminology: Use correct terminology for plant anatomy.
*   Petals: Describe edges (serrated, smooth, ruffled), texture (velvety, waxy, delicate), and arrangement (overlapping, sparse, symmetrical).
*   Center: Use terms like "stamen", "pistil", "anthers", "pollen", "cone", or "disk" when visible.
*   Leaves/Stem: Describe shape (lance-shaped, oval), arrangement, and surface (glossy, hairy, thorny).

Color and Texture: Be specific about colors. Do not just say "pink"; use "pale pink fading to white at the edges", "vibrant magenta", or "speckled purple". Describe patterns like "veining", "spots", "stripes", or "gradients".

Condition and State: Describe the physical state of the flower. Examples: "fully in bloom", "closed bud", "drooping petals", "withered edges", or "covered in dew droplets".

Environmental Description: Describe the setting strictly as seen. Examples: "green leafy background", "blurry garden setting", "studio black background", "natural sunlight", "dirt ground".

Camera Perspective and Style: Crucial for DiT training. Specify:
*   Shot Type: "Extreme close-up", "macro shot", "eye-level shot", "top-down view".
*   Focus: "Shallow depth of field", "bokeh background", "sharp focus", "soft focus".
*   Lighting: "Natural lighting", "harsh shadows", "dappled sunlight", "studio lighting".

Output Format: Output a single string containing the caption, without double quotes, using commas to separate phrases.</i>
</details>

## Training History and Configuration

Training utilizes **8-bit AdamW** and a **Cosine Schedule with 5% Warmup**. To achieve the best balance between structural coherence and sharp textures, the model underwent a two-stage training process: 1200 epochs using **MSE** (Global Structure), followed by 500 epochs of **L1** (Texture Sharpening).

| Configuration | Value | Purpose |
| :--- | :--- | :--- |
| **Loss** | **`MSE at 1e-4`** $\to$ **`L1 1e-4`** $\to$ **`L1 5e-5 for 200 Epochs`** | Initial training with MSE for stability; switched to L1 at E1200 for detail recovery. |
| **Batch Size** | **`12`** | Batch Size 12 was used over 16 because initially Self-Evaluation was supposed to be used. |
| **Shift Value** | **`1.0` (Uniform)** | Ensures a balanced training across all noise levels, critical for learning geometry on small datasets. |
| **Latent Norm** | **`0.0 Mean / 1.0 Std`** | Hardcoded identity normalization to preserve the relative channel relationships of the FLUX VAE. **Note:** Using a Mean and Std calculated from the dataset resulted in poor reconstruction with artifacts. |
| **EMA Decay** | **`0.999`** | Maintains a moving average of weights for smoother, higher-quality inference. |
| **Self-Evolution** | **`Disabled`** | Optional teacher-student distillation. (**Note:** Not used in this PoC to maintain baseline architectural clarity). |

### Loss & Fourier Gate Progression

| Loss Graph | Fourier Gate |
| :---: | :---: |
| ![Loss Graph](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/loss_curve.png?raw=true) | ![Fourier Gate](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2/blob/main/readme_assets/fourier_gate.png?raw=true) |

**Training Time Estimate:**
*   **GPU Time:** Approximately **3 hours** of total GPU compute time for 1500 epochs (RTX 5060 Ti 16GB).
*   **Project Time (Human):** 12 days of R&D, including hyperparameter tuning.

## Reproducibility

This repository is designed to be fully reproducible. The following data is included in the respective directories:
*   **Raw Dataset:** The original `.png` images and the **Qwen3-VL-4B-Instruct** generated and reviewed `.txt` captions.
*   **Cached Dataset:** The processed, tokenized, and VAE-encoded latents (`.pt` files).

## Repository File Breakdown

### Training & Core Scripts

| File | Purpose | Notes |
| :--- | :--- | :--- |
| **`train.py`** | Main training script. Supports EMA, Self-E, and Gradient Accumulation. | Includes automatic model compilation on Linux. |
| **`model.py`** | Defines `SingleStreamDiTV2` with Visual Fusion, Fourier Filters, and SwiGLU. | The core architecture definition. |
| **`config.py`** | Central configuration for paths, model dims, and hyperparameters. | All model settings are controlled here. |
| **`sanity_check.py`** | A utility to ensure the model can overfit to a single cached latent file. | Used for debugging architecture changes. |

### Utility & Preprocessing

| File | Purpose | Notes |
| :--- | :--- | :--- |
| **`preprocess.py`** | Prepares raw image/text data into cached `.pt` files using VAE and T5. | Run this before starting training. |
| **`calculate_cache_statistics.py`** | Analyzes cached latents to find Mean/Std for normalization settings. | **Note:** Use results with caution; defaults of 0.0/1.0 are often better. |
| **`debug_vae_pipeline.py`** | Tests the VAE reconstruction pipeline in float32 to isolate VAE issues. | Useful for troubleshooting color shifts. |
| **`check_cache.py`** | Decodes a single cached latent back to an image to verify preprocessing. | Fast integrity check. |
| **`generate_graph.py`** | Generates the loss curve visualization from the training CSV logs. | Creates `loss_curve.png`. |

### Inference & Data

| File | Purpose | Notes |
| :--- | :--- | :--- |
| **`inferenceNotebook.ipynb`** | Primary inference tool. Supports text-to-image with Euler/RK4. | Best for interactive testing. |
| **`samplers.py`** | Numerical integration steps for Euler and Runge-Kutta 4 (RK4). | Logic for the flow matching inference. |
| **`latents.py`** | Scaling and normalization logic for VAE latents. | Shared across preprocess, train, and inference. |
| **`dataset.py`** | Bucket-batching and RAM-caching dataset implementation. | Handles the training data pipeline. |