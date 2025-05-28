# Credit to Claude LLM for patch extraction and feature extraction code

"""
Main script for patch extraction and feature extraction
Coordinates the entire preprocessing pipeline
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from rich.console import Console
from rich.progress import Progress

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.patch_extractor import KidneyPatchExtractor
from preprocessing.uni_feature_extractor import UNIFeatureExtractor

console = Console()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_wsi_paths(wsi_dir: str) -> List[str]:
    """Get all WSI paths from the directory structure"""
    wsi_path = Path(wsi_dir)
    wsi_paths = []
    
    # Check if directory exists
    if not wsi_path.exists():
        console.print(f"[red]Error: WSI directory not found: {wsi_dir}[/red]")
        return []
    
    # Find all .svs files in subdirectories
    for subset_dir in ["wsi_set1", "wsi_set2", "wsi_set3"]:
        subset_path = wsi_path / subset_dir
        if subset_path.exists():
            svs_files = list(subset_path.glob("*.svs"))
            wsi_paths.extend([str(f) for f in svs_files])
            console.print(f"Found {len(svs_files)} WSIs in {subset_dir}")
    
    console.print(f"Total WSIs found: {len(wsi_paths)}")
    return sorted(wsi_paths)


def validate_wsi_labels(wsi_paths: List[str], labels_file: str) -> Dict[str, Any]:
    """Validate that all WSIs have corresponding labels"""
    # Load labels
    labels_df = pd.read_csv(labels_file)
    console.print(f"Loaded labels for {len(labels_df)} slides")
    
    # Get WSI names
    wsi_names = [Path(wsi_path).name for wsi_path in wsi_paths]
    label_names = set(labels_df['filename'].tolist())
    
    # Find matches and mismatches
    matched_wsis = []
    missing_labels = []
    
    for wsi_path, wsi_name in zip(wsi_paths, wsi_names):
        if wsi_name in label_names:
            matched_wsis.append(wsi_path)
        else:
            missing_labels.append(wsi_name)
    
    # Report validation results
    console.print(f"WSIs with labels: {len(matched_wsis)}")
    if missing_labels:
        console.print(f"[yellow]WSIs without labels ({len(missing_labels)}): {missing_labels[:5]}...[/yellow]")
    
    # Get label statistics
    target_column = "T"  # Tubulitis score
    if target_column in labels_df.columns:
        valid_labels = labels_df[target_column].dropna()
        console.print(f"Tubulitis score distribution:")
        for score in sorted(valid_labels.unique()):
            count = (valid_labels == score).sum()
            console.print(f"  Score {score}: {count} slides")
    
    return {
        "matched_wsis": matched_wsis,
        "missing_labels": missing_labels,
        "labels_df": labels_df
    }


def run_patch_extraction(config: Dict[str, Any], wsi_paths: List[str]) -> Dict[str, Any]:
    """Run patch extraction for all WSIs"""
    console.print("\n[bold blue]Starting patch extraction...[/bold blue]")
    
    # Initialize patch extractor
    extractor = KidneyPatchExtractor(config)
    
    # Extract patches
    output_dir = config['data']['output_dir']
    results = extractor.process_wsi_list(wsi_paths, output_dir)
    
    # Print summary
    summary = results['summary']
    console.print(f"\n[green]Patch extraction complete![/green]")
    console.print(f"Processed slides: {summary['successful_slides']}/{summary['total_slides_processed']}")
    console.print(f"Total patches extracted: {summary['total_patches_extracted']}")
    console.print(f"Average patches per slide: {summary['average_patches_per_slide']:.1f}")
    
    if results['failed_slides']:
        console.print(f"[yellow]Failed slides: {len(results['failed_slides'])}[/yellow]")
        for failed in results['failed_slides'][:3]:  # Show first 3 failures
            console.print(f"  {failed['slide_name']}: {failed['status']}")
    
    return results


def run_feature_extraction(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run UNI feature extraction"""
    console.print("\n[bold blue]Starting feature extraction...[/bold blue]")
    
    # Initialize feature extractor
    extractor = UNIFeatureExtractor(config)
    
    # Extract features
    patch_dir = config['data']['output_dir']
    features_dir = config['data']['features_dir']
    
    results = extractor.process_all_slides(patch_dir, features_dir)
    
    # Print summary
    summary = results['summary']
    console.print(f"\n[green]Feature extraction complete![/green]")
    console.print(f"Processed slides: {summary['successful_slides']}/{summary['total_slides_processed']}")
    console.print(f"Total features extracted: {summary['total_features_extracted']}")
    console.print(f"Feature dimension: {summary['feature_dimension']}")
    
    return results


def create_data_splits(config: Dict[str, Any], validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create train/validation/test splits"""
    console.print("\n[bold blue]Creating data splits...[/bold blue]")
    
    labels_df = validation_results['labels_df']
    matched_wsis = validation_results['matched_wsis']
    
    # Filter labels for matched WSIs
    wsi_names = [Path(wsi_path).name for wsi_path in matched_wsis]
    filtered_labels = labels_df[labels_df['filename'].isin(wsi_names)].copy()
    
    # Get target column
    target_column = config['data']['target_column']
    filtered_labels = filtered_labels.dropna(subset=[target_column])
    
    # Split configuration
    split_config = config['data']['split']
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['val_ratio']
    test_ratio = split_config['test_ratio']
    random_seed = split_config['random_seed']
    
    # Stratified split based on tubulitis scores
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        filtered_labels,
        test_size=test_ratio,
        stratify=filtered_labels[target_column],
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        stratify=train_val_data[target_column],
        random_state=random_seed
    )
    
    # Create split dictionary
    splits = {
        'train': train_data['filename'].tolist(),
        'val': val_data['filename'].tolist(),
        'test': test_data['filename'].tolist()
    }
    
    # Print split statistics
    console.print(f"Data splits created:")
    for split_name, slide_names in splits.items():
        split_labels = filtered_labels[filtered_labels['filename'].isin(slide_names)][target_column]
        console.print(f"  {split_name}: {len(slide_names)} slides")
        for score in sorted(split_labels.unique()):
            count = (split_labels == score).sum()
            console.print(f"    Score {score}: {count} slides")
    
    # Save splits
    splits_dir = Path(config['data']['features_dir']) / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    import json
    splits_file = splits_dir / "data_splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    console.print(f"Data splits saved to: {splits_file}")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description="Extract patches and features from kidney biopsy WSIs")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--skip-patches", action="store_true", help="Skip patch extraction")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-splits", action="store_true", help="Skip data split creation")
    
    args = parser.parse_args()
    
    # Load configuration
    console.print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Get WSI paths
    console.print("\n[bold blue]Discovering WSI files...[/bold blue]")
    wsi_paths = get_wsi_paths(config['data']['wsi_dir'])
    
    if not wsi_paths:
        console.print("[red]No WSI files found. Exiting.[/red]")
        return
    
    # Validate WSI labels
    console.print("\n[bold blue]Validating WSI labels...[/bold blue]")
    validation_results = validate_wsi_labels(wsi_paths, config['data']['labels_file'])
    
    if not validation_results['matched_wsis']:
        console.print("[red]No WSIs with matching labels found. Exiting.[/red]")
        return
    
    # Use only WSIs with labels
    wsi_paths = validation_results['matched_wsis']
    
    # Run patch extraction
    if not args.skip_patches:
        patch_results = run_patch_extraction(config, wsi_paths)
    else:
        console.print("[yellow]Skipping patch extraction[/yellow]")
    
    # Run feature extraction
    if not args.skip_features:
        feature_results = run_feature_extraction(config)
    else:
        console.print("[yellow]Skipping feature extraction[/yellow]")
    
    # Create data splits
    if not args.skip_splits:
        splits = create_data_splits(config, validation_results)
    else:
        console.print("[yellow]Skipping data split creation[/yellow]")
    
    console.print("\n[bold green]Preprocessing pipeline complete![/bold green]")
    console.print("Ready for model training!")


if __name__ == "__main__":
    main() 