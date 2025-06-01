# Single-Stage Tubulitis Scoring Pipeline

A novel end-to-end/single-stage pipeline for automated Banff tubulitis scoring from whole slide images using the UNI foundation model and CLAM architecture.

## Overview

This pipeline takes whole slide images (WSIs) and directly predicts tubulitis scores without requiring intermediate segmentation or detection steps. It supports both **regression** (continuous scores 0.0-3.0) and **ordinal classification** (discrete classes T0, T1, T2, T3) approaches.

### Key Features:
- **UNI Foundation Model**: Pre-trained pathology-specific features
- **CLAM Architecture**: Clustering-constrained Attention Multiple Instance Learning
- **Dual Approaches**: Regression and ordinal classification
- **Unified Training Framework**: Single script with multiple training modes
- **Rigorous Evaluation**: Held-out test sets and cross-validation
- **Comprehensive Metrics**: All clinical metrics for both tasks
- **Configurable Sampling**: Whole-tissue vs tubular-focused patch extraction

## Pipeline Components

1. **Data Preprocessing** (`preprocessing/`)
   - Patch extraction from WSIs
   - UNI feature extraction
   - Data split generation

2. **Model Architecture** (`models/`)
   - CLAM implementation (shared base)
   - **CLAMRegressor**: Continuous score prediction (0.0-3.0)
   - **CLAMClassifier**: Ordinal classification (T0, T1, T2, T3)
   - UNI feature extractor integration
   - Loss functions and metrics

3. **Training Framework** (`training/`)
   - **Unified Training Script**: Single entry point for all training modes
   - **Enhanced CLAMRegressorTrainer**: Label smoothing and gradient clipping
   - **Multiple Training Modes**: Single model, ensemble, CV, CV with holdout
   - Model checkpointing and comprehensive metrics

4. **Evaluation** (`evaluation/`)
   - **Regression metrics**: MSE, MAE, R², Pearson correlation
   - **Classification metrics**: Accuracy, F1, Kappa, confusion matrix
   - **Ordinal-specific**: Adjacent accuracy, Kendall's tau
   - **Clinical metrics**: Binary classification (low vs high grade)
   - **Holdout evaluation**: Unbiased final performance assessment
   - Comprehensive visualization tools

5. **Configuration** (`configs/`)
   - `clam_regressor_training.yaml`: Regression model config
   - `clam_classifier_training.yaml`: Classification model config
   - Training hyperparameters and experiment settings

## Usage

### Regression Model (Continuous Scores)

#### Training Modes

```bash
# Default: Cross-validation with held-out test set (RECOMMENDED)
python train_regressor.py

# Cross-validation using all data (for development)
python train_regressor.py --mode cv --cv-folds 5

# Ensemble training with different random seeds
python train_regressor.py --mode ensemble --ensemble-size 5

# Single model training
python train_regressor.py --mode single

# Custom holdout size
python train_regressor.py --mode cv-holdout --holdout-size 20 --cv-folds 5
```

#### Evaluation

```bash
# Evaluate CV ensemble on held-out test set
python evaluate_regressor.py --cv-ensemble --checkpoints-dir checkpoints_regressor

# Evaluate single best model on held-out test set
python evaluate_regressor.py --holdout --model-path checkpoints_regressor/cv_fold_0/best_model.pth

# Evaluate all CV folds (development evaluation)
python evaluate_regressor.py --all-folds --checkpoints-dir checkpoints_regressor

# Evaluate specific fold
python evaluate_regressor.py --fold cv_fold_0 --checkpoints-dir checkpoints_regressor
```

### Classification Model (Ordinal Classes)

```bash
# Train classifier (similar modes available)
python train_classifier.py --config configs/clam_classifier_training.yaml

# Evaluate classifier
python evaluate_classifier.py --all-folds

# Cross-validation
python train_classifier.py --cv --cv-folds 5

# Ensemble training
python train_classifier.py --ensemble --num-models 5
```

### Advanced Configuration

```bash
# Custom config file
python train_regressor.py --config my_custom_config.yaml

# Different number of CV folds
python train_regressor.py --mode cv-holdout --cv-folds 10

# Larger ensemble
python train_regressor.py --mode ensemble --ensemble-size 10
```

## Training Workflow

### Recommended Workflow (Rigorous Evaluation)

1. **Training with Held-out Test Set**:
   ```bash
   python train_regressor.py --mode cv-holdout
   ```
   - Creates stratified held-out test set (18 slides by default)
   - Runs 5-fold CV on remaining development data
   - Saves holdout split for later evaluation

2. **Model Selection**:
   - Review CV results to select best performing fold
   - Or use ensemble of all folds for final evaluation

3. **Final Evaluation**:
   ```bash
   # Single best model
   python evaluate_regressor.py --holdout --model-path checkpoints_regressor/cv_fold_X/best_model.pth
   
   # CV ensemble (recommended)
   python evaluate_regressor.py --cv-ensemble --checkpoints-dir checkpoints_regressor
   ```

### Development Workflow (All Data)

1. **Cross-validation for Development**:
   ```bash
   python train_regressor.py --mode cv --cv-folds 5
   ```

2. **Evaluate All Folds**:
   ```bash
   python evaluate_regressor.py --all-folds
   ```

## Model Architectures

### Shared CLAM Base
- Feature extraction: UNI (1024-dim) → Linear (128-dim)
- Attention mechanism: Gated or standard attention
- Instance-level regularization: Top-k sampling with SVM loss

### CLAMRegressor (Enhanced)
- **Task**: Continuous score prediction
- **Output**: Single value, clamped to [0, 3]
- **Loss**: MSE with optional label smoothing
- **Training**: Gradient clipping, mixed precision support
- **Metrics**: MSE, MAE, R², Pearson correlation

### CLAMClassifier
- **Task**: Ordinal classification (T0 < T1 < T2 < T3)
- **Output**: 3 cumulative thresholds → 4 class probabilities
- **Loss**: Ordinal cross-entropy (respects ordering)
- **Metrics**: Accuracy, F1, Kappa, MAE, adjacent accuracy

## Key Differences: Regression vs Classification

| Aspect | Regression | Classification |
|--------|------------|----------------|
| **Output** | Continuous [0, 3] | Discrete {0, 1, 2, 3} |
| **Loss** | MSE + Label Smoothing | Ordinal Cross-Entropy |
| **Error Handling** | All errors equal | Distance-aware errors |
| **Clinical Interpretation** | Exact scores | Clear categories |
| **Metrics** | MSE, MAE, R² | Accuracy, F1, Kappa |
| **Best For** | Fine-grained scoring | Decision support |

## Evaluation Metrics

### Regression Metrics
- **MSE/MAE**: Standard regression metrics
- **RMSE**: Root mean squared error
- **R²**: Explained variance
- **Pearson Correlation**: Linear relationship strength
- **Spearman Correlation**: Rank correlation

### Classification Metrics
- **Accuracy**: Overall correctness
- **Macro/Weighted F1**: Balanced performance metrics
- **Quadratic Weighted Kappa**: Ordinal agreement
- **Adjacent Accuracy**: Predictions within ±1 class
- **Confusion Matrix**: Detailed error analysis

### Clinical Metrics (Both)
- **Exact Accuracy**: Perfect score matches (rounded for regression)
- **Adjacent Accuracy**: Predictions within ±1 score
- **Binary Classification**: Low-grade (T0,T1) vs High-grade (T2,T3)
- **Sensitivity**: Detection of high-grade tubulitis
- **Specificity**: Correct identification of low-grade cases

## Advanced Features

### Enhanced Training (Regression)
- **Label Smoothing**: Adds noise to targets for better generalization
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Faster training with automatic mixed precision
- **Stratified Splits**: Maintains class balance across folds

### Comprehensive Visualization
- **Scatter Plots**: Predictions vs ground truth
- **Error Analysis**: Distribution of prediction errors
- **Confusion Matrices**: Detailed classification performance
- **Class-wise Metrics**: Performance breakdown by tubulitis grade
- **Correlation Analysis**: Statistical relationship assessment

### Holdout Evaluation
- **Unbiased Assessment**: True held-out test set never seen during development
- **Ensemble Evaluation**: Average predictions from multiple CV folds
- **Statistical Analysis**: Confidence intervals and significance testing

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenSlide
- UNI model dependencies (Need to log in with HuggingFace token)
- scikit-learn, scipy (for metrics)
- matplotlib, seaborn (for visualization)
- See requirements.txt for full list

## Model Selection Guidance

### Use **Regression** when:
- You need precise continuous scores
- Fine-grained scoring is important
- Comparing with other continuous systems
- Working with expert annotations that include fractional scores

### Use **Classification** when:
- You need clear diagnostic categories
- Clinical decision-making support
- Ordinal relationships are important
- Small dataset (better generalization)
- Integration with existing categorical systems

### Training Mode Selection:

- **`cv-holdout`** (Default): For final model evaluation and publication
- **`cv`**: For development and hyperparameter tuning
- **`ensemble`**: When you need maximum performance
- **`single`**: For quick experiments or resource-constrained environments

Both models use identical preprocessing, feature extraction, and training procedures, making them directly comparable for your specific use case. 