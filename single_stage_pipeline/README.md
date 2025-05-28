# Single-Stage Tubulitis Scoring Pipeline

A novel end-to-end/single-stage pipeline for automated Banff tubulitis scoring from whole slide images using the UNI foundation model and CLAM architecture.

## Overview

This pipeline takes whole slide images (WSIs) and directly predicts tubulitis scores (0.0-3.0) without requiring intermediate segmentation or detection steps. It uses:

- **UNI Foundation Model**: Pre-trained pathology-specific features
- **CLAM Architecture**: Clustering-constrained Attention Multiple Instance Learning
- **Configurable Sampling**: Whole-tissue vs tubular-focused patch extraction

## Pipeline Components

1. **Data Preprocessing** (`preprocessing/`)
   - Patch extraction from WSIs
   - UNI feature extraction
   - Data split generation

2. **Model Architecture** (`models/`)
   - CLAM implementation
   - UNI feature extractor integration
   - Loss functions and metrics

3. **Training Framework** (`training/`)
   - Training loops
   - Validation and testing
   - Model checkpointing

4. **Evaluation** (`evaluation/`)
   - Performance metrics
   - Comparison with existing pipeline
   - Visualization tools

5. **Configuration** (`configs/`)
   - Model configurations
   - Training hyperparameters
   - Experiment settings

## Usage

```bash
# Extract patches and features
python preprocessing/extract_patches.py --config configs/patch_extraction.yaml

# Train model
python train_regressor.py

# Evaluate model
python evaluate_regressor.py --all-folds # for regressor

```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenSlide
- UNI model dependencies (Need to log in with HuggingFace token)
- See requirements.txt for full list 