
# run from the kidney-grader root directory
# python e2e_model_transMIL/train_transmil.py --config e2e_model_transMIL/configs/transmil_regressor_config.yaml
# python e2e_model_transMIL/train_transmil.py --config e2e_model_transMIL/configs/transmil_regressor_config.yaml --extract_features_only


import argparse
import yaml
import pandas as pd
from pathlib import Path
import sys
from typing import Dict

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))

from e2e_model_transMIL.training.trainer import TransMILTrainer
from e2e_model_transMIL.data.tubule_patch_extractor import create_tubule_extractor, analyze_storage_efficiency
from e2e_model_transMIL.data.data_utils import create_cv_splits


def load_config(config_path: str) -> Dict:
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_features_for_all_wsis(config: Dict) -> None:
    
    print("starting feature extraction for all WSIs...")
    
    labels_df = pd.read_csv(config['data']['labels_file'])
    
    wsi_base_dir = Path("../all_wsis")
    wsi_paths = []
    
    print(f"searching for WSIs in: {wsi_base_dir.absolute()}")
    
    for filename in labels_df['filename']:
        found = False
        for wsi_set_dir in wsi_base_dir.glob("wsi_set*"):
            wsi_path = wsi_set_dir / filename
            if wsi_path.exists():
                wsi_paths.append(str(wsi_path))
                print(f"found: {wsi_path}")
                found = True
                break
        
        if not found:
            print(f"warning: WSI not found: {filename}")
    
    print(f"found {len(wsi_paths)} WSIs for feature extraction")
    
    if len(wsi_paths) == 0:
        print("error: no WSI files found. please check the directory structure.")
        print(f"expected WSI base directory: {wsi_base_dir.absolute()}")
        print("expected subdirectories: wsi_set1, wsi_set2, wsi_set3")
        return
    
    extractor = create_tubule_extractor(config)
    
    features_dir = Path(config['data']['features_dir'])
    features_dir.mkdir(parents=True, exist_ok=True)
    
    results = extractor.batch_extract(wsi_paths, str(features_dir))
    
    print(f"feature extraction completed successfully!")
    print(f"processed: {len(results)} slides")
    total_features = sum(r['num_patches'] for r in results if r.get('num_patches') is not None)
    print(f"total features: {total_features}")
    
    analyze_storage_efficiency(config['data']['features_dir'])
    
    return results

def main():
    parser = argparse.ArgumentParser(description="train TransMIL model for tubulitis scoring")
    parser.add_argument("--config", type=str, required=True,
                       help="path to configuration file")
    parser.add_argument("--extract_features_only", action="store_true",
                       help="only extract features, don't train model")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"loaded configuration from {args.config}")
    
    print("\nkey configuration details:")
    print(f"model: {config['model']['name']}")
    print(f"feature_dim: {config['model']['feature_dim']}")
    print(f"hidden_dim: {config['model']['hidden_dim']}")
    print(f"max_patches: {config['model']['max_patches']}")
    print(f"learning_rate: {config['training']['lr']}")
    print(f"epochs: {config['training']['epochs']}")
    print(f"n_folds: {config['data']['n_folds']}")
    print(f"device: {config['hardware']['device']}")
    
    if args.extract_features_only:
        extract_features_for_all_wsis(config)
        return
    
    features_dir = Path(config['data']['features_dir'])
    if not features_dir.exists():
        print(f"features directory not found: {features_dir}")
        print("run with --extract_features_only first to extract features")
        return
    
    feature_files = list(features_dir.glob("*_tubule_features.h5"))
    print(f"found {len(feature_files)} feature files in {features_dir}")
    
    if len(feature_files) == 0:
        print("no feature files found. run feature extraction first")
        return
    
    trainer = TransMILTrainer(config)
    
    print("\nstarting cross-validation training...")
    cv_results = trainer.cross_validate()
    
    print("\n" + "="*60)
    print("cross-validation training completed!")
    print("="*60)
    
    cv_summary = cv_results['cv_summary']
    
    print(f"\nfinal results (mean ± std across {config['data']['n_folds']} folds):")
    print(f"val_mse: {cv_summary['val_mse_mean']:.4f} ± {cv_summary['val_mse_std']:.4f}")
    print(f"val_mae: {cv_summary['val_mae_mean']:.4f} ± {cv_summary['val_mae_std']:.4f}")
    print(f"val_r2: {cv_summary['val_r2_mean']:.4f} ± {cv_summary['val_r2_std']:.4f}")
    
    if 'val_discrete_accuracy_mean' in cv_summary:
        print(f"discrete_accuracy: {cv_summary['val_discrete_accuracy_mean']:.4f} ± {cv_summary['val_discrete_accuracy_std']:.4f}")
    if 'val_within_1_accuracy_mean' in cv_summary:
        print(f"within_1_accuracy: {cv_summary['val_within_1_accuracy_mean']:.4f} ± {cv_summary['val_within_1_accuracy_std']:.4f}")
    if 'val_pearson_correlation_mean' in cv_summary:
        print(f"pearson_correlation: {cv_summary['val_pearson_correlation_mean']:.4f} ± {cv_summary['val_pearson_correlation_std']:.4f}")
    
    print(f"\ncheckpoints and results saved to: {config['checkpoint']['save_dir']}")
    
    if config['visualization']['save_attention_heatmaps']:
        print(f"attention visualizations saved to: {config['visualization']['attention_output_dir']}")


if __name__ == "__main__":
    main() 