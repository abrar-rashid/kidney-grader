# TransMIL for Tubulitis Scoring

A state-of-the-art transformer-based multiple instance learning (TransMIL) approach for automated tubulitis scoring in kidney biopsies. This model focuses specifically on tubule regions using segmentation masks and provides comprehensive attention visualizations.

## Key Features

- **Tubule-focused sampling**: Uses segmentation masks to extract patches only from tubule regions
- **Advanced attention mechanism**: Bidirectional transformer with 2D positional encoding for spatial relationships
- **Comprehensive visualizations**: Attention heatmaps overlaid on original WSIs with top attended patches
- **Robust training**: Optimized for small datasets with cross-validation, mixed precision, and advanced regularization
- **UNI feature extraction**: Leverages state-of-the-art universal histology features

## Architecture

### Model Components
- **TransMILAggregator**: Core transformer with bidirectional attention
- **2D Positional Encoding**: Spatial relationship modeling for patch coordinates  
- **Instance-level Pseudo-labeling**: Improves attention focus on relevant tubule regions
- **Multi-scale Loss**: Bag-level regression + instance classification + attention regularization

### Key Innovations
- **Tubule-guided sampling**: Only processes patches with >30% tubule content
- **Attention regularization**: Encourages sparse, focused attention on inflammatory regions
- **Cross-validation**: 5-fold CV with 18 WSI held-out test set
- **Memory optimization**: Supports up to 10000 patches per WSI using 24-48GB GPU memory

## Installation

```bash
# ensure you have the parent kidney-grader environment activated
cd kidney-grader/e2e_model_transMIL

# install additional dependencies if needed
pip install tiffslide h5py matplotlib seaborn scikit-learn scipy
```

### UNI Model Access (Optional but Recommended)

The TransMIL implementation uses the UNI foundation model for feature extraction, which provides state-of-the-art histopathology features. To use UNI:

1. **Request Access**: Visit [https://huggingface.co/MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) and request access
2. **Login**: Run `huggingface-cli login` and provide your token
3. **Automatic Usage**: Once approved, UNI will be used automatically

**Fallback**: If UNI access is not available, the system automatically falls back to ImageNet ViT-Large, which still provides excellent 1024-dimensional features suitable for histopathology analysis.

## Usage

### 1. Feature Extraction

First, extract tubule-focused features from all WSIs:

```bash
python train_transmil.py --config configs/transmil_regressor_config.yaml --extract_features_only
```

This will:
- Load segmentation model
- Extract tissue patches from all WSIs
- Filter patches by tubule content (>30% tubule pixels)
- Extract UNI features for tubule patches
- Save features to `./data/tubule_features/`

### 2. Model Training

Run cross-validation training:

```bash
python train_transmil.py --config configs/transmil_regressor_config.yaml
```

Training features:
- 5-fold cross-validation with stratified splits
- Mixed precision training for memory efficiency  
- Warmup learning rate schedule
- Early stopping with patience
- Attention visualizations saved during validation

### 3. Inference

#### Single WSI Prediction
```bash
python inference_transmil.py \
    --checkpoint checkpoints_transmil/fold_0_best.pth \
    --config configs/transmil_regressor_config.yaml \
    --wsi_path path/to/slide.svs \
    --output_dir results/single_prediction/
```

#### Batch Prediction
```bash
# create wsi_list.txt with one WSI path per line
python inference_transmil.py \
    --checkpoint checkpoints_transmil/fold_0_best.pth \
    --config configs/transmil_regressor_config.yaml \
    --wsi_list wsi_paths.txt \
    --output_dir results/batch_predictions/
```

## Configuration

Key parameters in `configs/transmil_regressor_config.yaml`:

```yaml
model:
  feature_dim: 1024      # UNI feature dimension
  hidden_dim: 256        # transformer hidden size (reduced for small dataset)
  num_layers: 2          # transformer depth
  num_heads: 8           # multi-head attention
  max_patches: 10000      # maximum patches per WSI

training:
  lr: 0.0001            # learning rate
  epochs: 200           # training epochs
  bag_loss_weight: 0.7  # bag-level loss weight
  instance_loss_weight: 0.25  # instance pseudo-labeling weight
  attention_reg_weight: 0.05  # attention regularization

data:
  min_tubule_ratio: 0.3  # minimum tubule content for patch inclusion
  max_patches: 10000      # memory-optimized for 24-48GB GPU
```

## Attention Visualizations

The model generates comprehensive attention visualizations:

### 1. WSI Attention Heatmap
- Attention weights overlaid on original WSI
- Top-10 patches highlighted with bounding boxes
- Heat map showing tubule region attention

### 2. Attention Analysis
- Attention weight distribution histograms
- Cumulative attention plots
- Spatial attention scatter plots
- Statistical summaries

### 3. Top Patches Visualization  
- Highest attended patch locations
- Attention weight values
- Coordinate information

## Expected Performance

Based on the small dataset (93 WSIs) and cross-validation:

- **MSE**: ~0.5-0.8 (tubulitis score range 0-3)
- **MAE**: ~0.4-0.6  
- **Discrete Accuracy**: ~60-75%
- **Within-1 Accuracy**: ~85-95%
- **Pearson Correlation**: ~0.6-0.8

## Model Architecture Details

### TransMIL Aggregator
```python
- Input projection: 1024 → 256 dimensions
- 2D positional encoding for spatial relationships
- 2 bidirectional transformer layers
- Class token for bag-level representation
- Instance classifier for pseudo-labeling
```

### Loss Components
1. **Bag Loss** (0.7): Huber loss for robust regression
2. **Instance Loss** (0.25): Pseudo-labeling based on attention + bag labels  
3. **Attention Reg** (0.05): Entropy + L1 for sparse attention

### Training Strategy
- **Warmup**: 10 epochs linear warmup
- **Scheduler**: Cosine annealing with restarts
- **Regularization**: Dropout (0.3), weight decay (0.02)
- **Gradient clipping**: Max norm 1.0
- **Mixed precision**: FP16 for memory efficiency

## File Structure

```
e2e_model_transMIL/
├── models/
│   ├── attention_modules.py      # Core attention mechanisms
│   ├── transmil_regressor.py     # Main model class
│   └── transmil_backbone.py      # Backbone architecture
├── data/
│   ├── tubule_patch_extractor.py # Tubule-guided patch extraction
│   ├── tubule_dataset.py         # Dataset and dataloader
│   └── data_utils.py             # Cross-validation and visualization
├── training/
│   ├── trainer.py                # Main training loop
│   ├── inference.py              # Inference with visualization
│   └── metrics.py                # Tubulitis-specific metrics
├── configs/
│   └── transmil_regressor_config.yaml
├── train_transmil.py             # Training script
├── inference_transmil.py         # Inference script
└── README.md
```

## Troubleshooting

### Memory Issues
- Reduce `max_patches` in config (try 2000-3000)
- Reduce `batch_size` in feature extraction
- Use smaller `hidden_dim` (128 or 192)

### Feature Extraction Fails
- Check segmentation model path in config
- Ensure WSI paths are correct
- Verify sufficient disk space for features

### Poor Attention Quality
- Increase `attention_reg_weight` for sparser attention
- Adjust `min_tubule_ratio` threshold
- Check segmentation mask quality

### Training Instability
- Increase warmup epochs
- Reduce learning rate
- Increase gradient clipping threshold

## Advanced Usage

### Custom Segmentation Integration
Modify `segment.py` to enhance tubule detection:
```python
# add to segmentation/segment.py
def enhanced_tubule_detection(mask, inflammation_threshold=0.1):
    # detect tubules with inflammatory infiltrates
    # your enhanced detection logic here
    return enhanced_mask
```

### Ensemble Predictions
```python
# load multiple fold models
models = [load_model(f"fold_{i}_best.pth") for i in range(5)]

# ensemble prediction
predictions = [model.predict(features, coords) for model in models]
ensemble_score = np.mean([p['predicted_score'] for p in predictions])
```

This TransMIL implementation represents a state-of-the-art approach specifically designed for tubulitis scoring with excellent attention visualization capabilities and robust performance on small datasets. 