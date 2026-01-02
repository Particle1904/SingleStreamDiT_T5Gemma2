# Single-Stream DiT (Proof-of-Concept)

This repository contains the codebase for a Single-Stream Diffusion Transformer (DiT) Proof-of-Concept, heavily inspired by modern architectures like **Z-Image** and **Lumina Image 2**.

The primary objective was to demonstrate the feasibility and training stability of coupling the high-fidelity **EQ-SDXL-VAE** with the powerful **T5Gemma2** text encoder for image generation on consumer-grade hardware (NVIDIA RTX 5060 Ti 16GB).

**Note on Final Checkpoint:** The final, best-performing **EMA checkpoint (Epoch 1200)** is uploaded to [Hugging Face and is linked separately from this repository](https://huggingface.co/Crowlley/SingleStreamDiT_T5Gemma2-EQ-VAE/tree/main).

## Project Overview

### How it started
![How it started](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2-EQ-VAE/blob/main/readme_assets/bored.png?raw=true)

### Verification and Result Comparison

| Cached Latent Verification | Final Generated Sample (RK4, Epoch 1200 EMA) |
| :---: | :---: |
| ![Cache Verification](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2-EQ-VAE/blob/main/readme_assets/cache_verification.png?raw=true) | ![Generated Sample](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2-EQ-VAE/blob/main/readme_assets/sample_rk4_cfg1.5.png?raw=true) |

## Core Models and Architecture

| Component | Model ID / Function | Purpose |
| :--- | :--- | :--- |
| **Generator** | `SingleStreamDiTV2` | Custom Diffusion Transformer model that predicts the velocity vector ($v$) along the flow trajectory. |
| **Text Encoder** | `google/t5gemma-2-1b-1b` | Used to generate rich, high-dimensional text embeddings (1152 dimensions) for conditional guidance (CFG). |
| **VAE** | `KBlueLeaf/EQ-SDXL-VAE` | A high-quality SDXL-compatible VAE used for latent compression and reconstruction. Crucial for handling fine details. |
| **Training Method** | Flow Matching (V-Prediction) | The model is trained to predict the vector field that transforms noise ($x_0$) to the clean latent ($x_1$). |

## Training Progression

| Early Epoch (Epoch 10) | Final Epoch (Epoch 1200, RAW) | Full Progression (GIF) |
| :---: | :---: | :---: |
| ![Epoch10](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2-EQ-VAE/blob/main/readme_assets/epoch_10.png?raw=true) | ![Epoch1200](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2-EQ-VAE/blob/main/readme_assets/epoch_1200.png?raw=true) | ![Epochs over time](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2-EQ-VAE/blob/main/readme_assets/epochsOverTime.gif?raw=true) |

## Data Curation and Preprocessing

The model was trained on a small, curated dataset of **200 images** (10 categories of flowers with 20 images each).

| Component | Tool / Method | Purpose / Detail |
| :--- | :--- | :--- |
| **Pre/Post-processing of Dataset** | **[Dataset Tools](https://github.com/Particle1904/DatasetHelpers)** | Used to resize images to 512x512 and to Edit the Qwen3-VL captions. |
| **Captioning** | **Qwen3-VL-4B-Instruct** | The dataset was captioned using a specialized visual language model with a strict botanical system instruction. This ensured captions contained precise details on **texture (waxy, serrated), plant anatomy (stamen, pistil), lighting (shallow depth of field), and shot type (macro shot)**. |
| **Data Encoding** | `preprocess.py` | Encodes images to latents via EQ-VAE and text via T5Gemma2, applying bucketing and horizontal flip augmentation (for Epochs 0-600). |

## Training History and Final Configuration

Training utilized a **Cosine Annealing Learning Rate Scheduler** across all epochs to facilitate steady convergence, starting with $1e-4$ and ending at $1e-5$.

| Epoch Range | Loss Function | Learning Rate (Start $\to$ End) | Key Features | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **0 - 600** | Mean Squared Error (MSE) | $1e-4 \to \sim 5e-5$ | Horizontal Flip Augmentation **Enabled**. | Fast convergence on shape, but resulted in an undesirable "waxy" finish. At Epoch 600, the **Horizontal Flip Augmentation was removed** due to it causing artifacts (e.g., generating two flower stems). |
| **601 - 900** | L1 Loss (MAE) | $5e-5 \to \sim 2e-5$ | Horizontal Flip **Disabled**. Switched to L1 Loss. | Modest improvement in sharpness and clarity. |
| **901 - 1200** | L1 Loss (MAE) | $5e-5 \to 1e-5$ | Horizontal Flip **Disabled**. **Introduced EMA** + Latent Normalization to $\text{Std} \approx 1.0$. | **Optimal result.** Eliminated "waxy" look, successfully recovering and sharpening the high-frequency textural details. |

### Loss Progression

![Loss Graph](https://github.com/Particle1904/SingleStreamDiT_T5Gemma2-EQ-VAE/blob/main/readme_assets/loss_curve.png?raw=true)

**Training Time Estimate:**
*   **GPU Time:** Approximately **5 hours** of total GPU compute time for 1200 epochs (based on an average epoch time of $\sim 14$ seconds).
*   **Project Time (Human):** The overall development and hyperparameter tuning project took approximately 2-3 days.

## Reproducibility

This repository is designed to be fully reproducible. The following data is included in the respective directories:
*   **Raw Dataset:** The original `.png` images and the **Qwen3-VL-4B-Instruct** generated `.txt` captions.
*   **Cached Dataset:** The processed, tokenized, and VAE-encoded latents (`.pt` files).
*   **Training Artifacts:** All checkpoint samples (from Epoch 10 to 1200) and all training log files (split by epoch range and a combined 0-1200 file).

### Key Configuration in `train_v3_ema.py`

| Configuration | Value | Purpose |
| :--- | :--- | :--- |
| **Loss** | `F.l1_loss(pred, target_v)` | Uniform L1/MAE loss, which proved superior to MSE for generating fine details. |
| **Latent Norm** | `x_1 = x_1 / 1.1908` | Normalizes latents to $\text{Std} \approx 1.0$ for optimal neural network stability. |
| **EMA Decay** | `EMA_DECAY = 0.999` | Ensures a stable, high-quality checkpoint is saved, preventing weight oscillation. **The final result uses the EMA weights.** |

## Repository File Breakdown

This section details the purpose and configurable parameters of each primary Python file.

### Training Scripts

| File | Purpose | Key Configs | Notes |
| :--- | :--- | :--- | :--- |
| **`train_v3_ema.py`** | **Final, Optimal Training Script.** Uses L1 Loss, Latent Normalization, Cosine LR Annealing, and **EMA**. | `RESUME_FROM`, `START_EPOCH`, `BATCH_SIZE`, `LEARNING_RATE`, `EMA_DECAY` | This is the recommended script for any new training runs. |
| **`train_v2_l1.py`** | Archive: Training script for epochs 601-900 (L1 Loss only). | `RESUME_FROM`, `START_EPOCH` | **DEPRECATED.** |
| **`train.py`** | Archive: Initial training script for epochs 0-600 (MSE Loss, basic, **with flip augmentation**). | `RESUME_FROM`, `START_EPOCH` | **DEPRECATED.** |
| **`train_overfit.py`** | A sanity check utility to ensure the model can overfit to a single data point. | `TARGET_FILE`, `STEPS` | For debugging architecture changes only. |

### Utility & Preprocessing

| File | Purpose | Key Configs | Notes |
| :--- | :--- | :--- | :--- |
| **`preprocess.py`** | Prepares the raw image/text data into cached `.pt` files. Encodes images to latents via EQ-VAE and text via T5Gemma2. | `DATASET_DIR`, `OUTPUT_DIR`, `BUCKETS` | Must be run once before training. Includes image-flipping data augmentation (used only in Epochs 0-600 training). |
| **`calculate_cache_statistics.py`** | Analyzes all cached `.pt` files to report the dataset's Mean and Standard Deviation. | `CACHE_DIR` | **CRITICAL** for determining the `LATENT_STD_SCALE` and `LATENT_OFFSET` used in training/inference. |
| **`check_cache.py`** | Decodes a single cached latent file back into an image using the VAE to verify the preprocessing integrity. | `TARGET_FILE`, `VAE_ID` | Quick sanity check. |

### Inference Scripts

| File | Purpose | Key Configs | Notes |
| :--- | :--- | :--- | :--- |
| **`inference_unified.py`** | **The main inference script.** Supports both **text-to-image** and **file-to-image** generation using either Euler or RK4 sampling. | `FILENAME` (for checkpoint), `INPUT_MODE`, `SAMPLER`, `GUIDANCE_SCALE`, `PROMPT`, `HEIGHT`, `WIDTH` | Recommended for all new generations. Integrates all necessary scaling factors. |
| **`inference_euler.py`** | Archive: Old script for file-to-image inference with Euler sampling. | N/A | **DEPRECATED.** |
| **`inference_rk4.py`** | Archive: Old script for file-to-image inference with RK4 sampling. | N/A | **DEPRECATED.** |
| **`text_inference.py`** | Archive: Old script for text-to-image inference with Euler sampling. | N/A | **DEPRECATED.** |
