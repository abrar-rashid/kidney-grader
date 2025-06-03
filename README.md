# KidneyGrader: Automated Banff Tubulitis Scoring Pipeline

An end-to-end computational pathology pipeline for automated Banff tubulitis scoring in kidney biopsy whole slide images (WSIs).

## Table of Contents

- [Installation](#installation)
- [Model Checkpoints](#model-checkpoints)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Parameters](#parameters)
- [Output Structure](#output-structure)
- [Batch Processing](#batch-processing)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for inference)
- 16GB+ RAM (for large WSI processing)

### 1. Clone Repository
```bash
git clone <repository-url>
cd kidney-grader
```

### 2. Create Virtual Environment
```bash
python -m venv .env
source .env/bin/activate  # Linux/Mac
# or
.env\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python main.py --help
```

## Model Checkpoints

The pipeline requires pre-trained models for segmentation and detection:

### Required Directory Structure:
```
checkpoints/
├── segmentation/
│   └── kidney_grader_unet.pth          # Our U-Net model for semantically segmenting tubules, glomeruli, arteries and indeterminate vessels
└── detection/
    └── inflammatory_cell_detector/     # Inflammatory cell detection models (InstanSeg and TiaKong)
        ├── model_weights.pth
        └── config.json
```

### To Download Models:
```bash
python download_checkpoints.py
```

## Pipeline Overview

The multi-stage version of the KidneyGrader pipeline consists of 3 main stages:

1. **Stage 1 - Segmentation**: Identify tubule instances in WSI
2. **Stage 2 - Detection**: Detect inflammatory cells throughout the WSI  
3. **Stage 3 - Grading**: Count cells per tubule and assign Banff tubulitis grade

## Quick Start

### Process a Single WSI:
```bash
python main.py \
    --input_path /path/to/slide.svs \
    --output_dir results \
    --stage all \
    --prob_thres 0.50
```

### Process Multiple WSIs:
```bash
python run_batch_pipeline.py \
    --input_dir /path/to/wsi/directory \
    --output_dir batch_results \
    --stage all \
    --prob_thres 0.50
```

## Usage

### Basic Syntax:
```bash
python main.py --input_path <WSI_PATH> --output_dir <OUTPUT_DIR> [OPTIONS]
```

### Required Arguments:
- `--input_path`: Path to WSI file (.svs, .tif, .tiff, .ndpi, etc.)
- `--output_dir`: Directory where results will be saved

### Optional Arguments:

#### **Pipeline Control:**
- `--stage {1,2,3,all,segment,detect,grade}`: Which stage(s) to run (default: all)
- `--force`: Recompute all stages even if outputs exist
- `--visualise`: Generate visualization overlays
- `--update_summary`: Update summary files after processing

#### **Model Parameters:**
- `--model_path`: Path to segmentation model (default: checkpoints/segmentation/kidney_grader_unet.pth)
- `--prob_thres`: Probability threshold for inflammatory cell filtering (default: 0.50)

#### **Custom Inputs (Advanced):**
- `--detection_json`: Use custom inflammatory cell detection JSON
- `--instance_mask_class1`: Use custom tubule segmentation mask

#### **Summary Generation:**
- `--summary_only`: Only regenerate summary files without processing WSIs

## Pipeline Stages

### Stage 1: Segmentation (`--stage 1` or `--stage segment`)
- **Input**: WSI file
- **Output**: Instance masks for tubules
- **Location**: `individual_reports/{wsi_name}/segmentation/`

### Stage 2: Detection (`--stage 2` or `--stage detect`)  
- **Input**: WSI file
- **Output**: Inflammatory cell coordinates with confidence scores
- **Location**: `individual_reports/{wsi_name}/detection/`

### Stage 3: Grading (`--stage 3` or `--stage grade`)
- **Input**: Segmentation masks + detection coordinates
- **Output**: Cell counts per tubule + Banff tubulitis grade
- **Location**: `individual_reports/{wsi_name}/grading/{param_tag}/`

### Multiple Stages:
```bash
# Run stages 2 and 3 only
python main.py --input_path slide.svs --stage 2,3

# Run all stages
python main.py --input_path slide.svs --stage all
```

## Parameters

### Probability Threshold (`--prob_thres`)
Controls which inflammatory cell detections to include based on model confidence:

- **0.30**: Include more cells (higher sensitivity, lower specificity)
- **0.50**: Balanced approach (default)
- **0.80**: Include only high-confidence cells (lower sensitivity, higher specificity)

**Example**: Compare different thresholds
```bash
python main.py --input_path slide.svs --prob_thres 0.30 --output_dir results_p030
python main.py --input_path slide.svs --prob_thres 0.50 --output_dir results_p050  
python main.py --input_path slide.svs --prob_thres 0.80 --output_dir results_p080
```

### Force Recomputation (`--force`)
By default, the pipeline skips stages if outputs already exist. Use `--force` to recompute:

```bash
# Recompute everything
python main.py --input_path slide.svs --force

# Recompute only grading with new threshold
python main.py --input_path slide.svs --stage grade --prob_thres 0.60 --force
```

### Visualization (`--visualise`)
Generate overlay images showing tubules and inflammatory cells:

```bash
python main.py --input_path slide.svs --visualise
```

**Output**: `individual_reports/{wsi_name}/grading/{param_tag}/visualization/`

## Output Structure

```
results/
├── individual_reports/
│   └── {wsi_name}/
│       ├── segmentation/
│       │   ├── {wsi_name}_full_instance_mask_class1.tiff
│       │   └── {wsi_name}_semantic_mask.tiff
│       ├── detection/
│       │   └── detected-inflammatory-cells.json
│       └── grading/
│           └── p050/                    # Parameter-specific results
│               ├── quantification.json  # Detailed cell counts & statistics
│               ├── tubule_counts.csv   # Per-tubule data
│               ├── grading_report.json # Banff grade & evaluation
│               └── visualization/      # Overlay images (if --visualise)
├── summary/
│   ├── aggregated_scores.csv          # All results combined
│   ├── evaluation_metrics.csv         # Performance metrics
│   └── version_YYYYMMDD_HHMMSS/       # Timestamped backups
└── logs/
    └── pipeline.log                   # Detailed processing logs
```

### Key Output Files:

#### **Quantification JSON** - Detailed cell counting data:
```json
{
  "wsi_name": "slide_001",
  "total_inflam_cells": 15420,
  "total_tubules": 3247,
  "mean_cells_per_tubule": 4.75,
  "per_tubule_inflam_cell_counts": {"tubule_1": 12, "tubule_2": 8, ...},
  "summary_stats": {...}
}
```

#### **Grading Report** - Clinical scoring results:
```json
{
  "wsi_name": "slide_001", 
  "tubulitis_score_predicted": "t2",
  "tubulitis_score_ground_truth": "t1",
  "score_difference": 1.0,
  "correct_category": false,
  "prob_thres": 0.50
}
```

#### **Tubule Counts CSV** - Per-tubule analysis:
```csv
tubule_id,x,y,cell_count
1001,1250.5,980.2,12
1002,1340.1,1020.8,8
1003,1180.9,1150.3,15
```

## Batch Processing

Process multiple WSIs efficiently using the batch script:

### Basic Batch Processing:
```bash
python run_batch_pipeline.py \
    --input_dir /path/to/wsi/directory \
    --output_dir batch_results
```

### Advanced Batch Options:
```bash
python run_batch_pipeline.py \
    --input_dir /path/to/slides \
    --output_dir batch_results \
    --stage grade \
    --prob_thres 0.60 \
    --max_workers 4 \
    --pattern "*case*" \
    --limit 10 \
    --timeout 7200 \
    --update_summary
```

#### **Batch Parameters:**
- `--max_workers`: Number of parallel processes (default: 1)
- `--pattern`: Only process files matching pattern (e.g., "*case1*")  
- `--limit`: Process only first N files (useful for testing)
- `--timeout`: Timeout per WSI in seconds (default: 3600)

## Examples

### 1. Full Pipeline with Visualization:
```bash
python main.py \
    --input_path kidney_biopsy_001.svs \
    --output_dir results \
    --stage all \
    --prob_thres 0.50 \
    --visualise \
    --update_summary
```

### 2. Grading Only (with existing segmentation/detection):
```bash
python main.py \
    --input_path kidney_biopsy_001.svs \
    --output_dir results \
    --stage grade \
    --prob_thres 0.75 \
    --force
```

### 3. Parameter Sweep:
```bash
for threshold in 0.30 0.50 0.70 0.90; do
    python main.py \
        --input_path kidney_biopsy_001.svs \
        --output_dir results \
        --stage grade \
        --prob_thres $threshold \
        --force
done

# Generate combined summary
python main.py --input_path dummy --output_dir results --summary_only
```

### 4. Batch Processing with Custom Models:
```bash
python run_batch_pipeline.py \
    --input_dir /data/kidney_slides \
    --output_dir /results/batch_v2 \
    --model_path /models/custom_segmentation.pth \
    --prob_thres 0.60 \
    --max_workers 2 \
    --visualise \
    --update_summary
```

### 5. Using Custom Inputs:
```bash
# Use pre-computed segmentation and detection
python main.py \
    --input_path slide.svs \
    --output_dir results \
    --stage grade \
    --instance_mask_class1 /path/to/custom_tubules.tiff \
    --detection_json /path/to/custom_detections.json
```

## Ground Truth Integration

If you have ground truth Banff scores, create a `banff_scores.csv` file:

```csv
filename,T
kidney_biopsy_001.svs,1
kidney_biopsy_002.svs,2
kidney_biopsy_003.svs,0
```

The pipeline will automatically:
- Compare predictions to ground truth
- Calculate accuracy metrics
- Generate evaluation reports in summary files

## Troubleshooting

### Common Issues:

#### **CUDA/GPU Issues:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode if needed
export CUDA_VISIBLE_DEVICES=""
```

#### **Memory Issues:**
- Reduce batch size in detection models
- Use smaller probability thresholds
- Process slides sequentially (`--max_workers 1`)

#### **Missing Dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Install specific packages
pip install tiffslide openslide-python
```

#### **File Permission Issues:**
```bash
# Fix permissions
chmod -R 755 checkpoints/
chmod +x main.py run_batch_pipeline.py
```

#### **Model Loading Errors:**
- Verify checkpoint paths exist
- Check model file integrity
- Ensure compatible PyTorch version

