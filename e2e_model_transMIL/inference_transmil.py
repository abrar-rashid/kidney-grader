# from kidney-grader root directory:
# singe: python e2e_model_transMIL/inference_transmil.py --checkpoint e2e_model_transMIL/checkpoints_transmil/fold_0_best.pth --config e2e_model_transMIL/configs/transmil_regressor_config.yaml --wsi_path path/to/slide.svs --output_dir results/
# batch: python e2e_model_transMIL/inference_transmil.py --checkpoint e2e_model_transMIL/checkpoints_transmil/fold_0_best.pth --config e2e_model_transMIL/configs/transmil_regressor_config.yaml --wsi_list wsi_paths.txt --output_dir results/


import argparse
import yaml
from pathlib import Path
import sys
from typing import List
import pandas as pd

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))  # kidney-grader root

from training.inference import TransMILInference


def load_wsi_list(wsi_list_path: str) -> List[str]:
    
    with open(wsi_list_path, 'r') as f:
        wsi_paths = [line.strip() for line in f if line.strip()]
    
    valid_paths = []
    for wsi_path in wsi_paths:
        if Path(wsi_path).exists():
            valid_paths.append(wsi_path)
        else:
            print(f"warning: WSI not found: {wsi_path}")
    
    return valid_paths


def main():
    parser = argparse.ArgumentParser(description="run inference with TransMIL model for tubulitis scoring")
    
    parser.add_argument('--config', type=str, required=True, 
                       help='path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='path to trained model checkpoint')
    parser.add_argument('--wsi_path', type=str, default=None,
                       help='path to single WSI for prediction')
    parser.add_argument('--wsi_list', type=str, default=None,
                       help='path to text file with list of WSI paths')
    parser.add_argument('--wsi_dir', type=str, default=None,
                       help='directory containing WSI files (.svs)')
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='output directory for results and visualizations')
    parser.add_argument('--visualise', action='store_true',
                       help='create attention heatmap visualizations (default: off)')
    
    parser.add_argument('--labels_csv', type=str, default='banff_scores.csv',
                       help='path to CSV file with ground truth labels (default: banff_scores.csv)')
    
    args = parser.parse_args()
    
    if not args.wsi_path and not args.wsi_list and not args.wsi_dir:
        print("error: must specify either --wsi_path, --wsi_list, or --wsi_dir")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"error: checkpoint not found: {checkpoint_path}")
        return
    
    labels_csv_path = args.labels_csv
    if not Path(labels_csv_path).is_absolute():
        labels_csv_path = str(Path.cwd() / labels_csv_path)
    
    print(f"initializing TransMIL inference model...")
    print(f"checkpoint: {checkpoint_path}")
    print(f"labels CSV: {labels_csv_path}")
    
    inference_model = TransMILInference(
        checkpoint_path=str(checkpoint_path),
        config=config,
        labels_csv_path=labels_csv_path
    )
    
    save_visualizations = args.visualise
    
    # single WSI prediction
    if args.wsi_path:
        print(f"\nrunning inference on single WSI: {args.wsi_path}")
        
        if not Path(args.wsi_path).exists():
            print(f"error: WSI not found: {args.wsi_path}")
            return
        
        result = inference_model.predict_single_wsi(
            wsi_path=args.wsi_path,
            output_dir=str(output_dir),
            save_visualizations=save_visualizations
        )
        
        print("\n" + "="*50)
        print("inference results")
        print("="*50)
        print(f"slide name: {result['slide_name']}")
        print(f"predicted tubulitis score: {result['predicted_score']:.3f}")
        print(f"number of tubule patches: {result['num_patches']}")
        print(f"max attention weight: {result['top_attention_values'][0]:.4f}")
        print(f"mean attention weight: {result['attention_weights'].mean():.4f}")
        print(f"attention std: {result['attention_weights'].std():.4f}")
        
        print(f"\ntop 5 attended patches:")
        for i in range(min(5, len(result['top_attention_indices']))):
            idx = result['top_attention_indices'][i]
            weight = result['top_attention_values'][i]
            coord = result['patch_coordinates'][idx]
            print(f"  rank {i+1}: patch {idx} at ({coord[0]:.0f}, {coord[1]:.0f}) - weight: {weight:.8f}")
        
        if save_visualizations:
            print(f"\nattention visualizations saved to: {output_dir}")
    
    # batch prediction
    elif args.wsi_list:
        print(f"\nrunning batch inference from WSI list: {args.wsi_list}")
        
        if not Path(args.wsi_list).exists():
            print(f"error: WSI list file not found: {args.wsi_list}")
            return
        
        wsi_paths = load_wsi_list(args.wsi_list)
        
        if not wsi_paths:
            print("error: no valid WSI paths found in list")
            return
        
        print(f"found {len(wsi_paths)} valid WSI paths")
        
        results_df = inference_model.predict_batch_wsis(
            wsi_paths=wsi_paths,
            output_dir=str(output_dir),
            save_individual_visualizations=save_visualizations
        )
        
        print("\n" + "="*50)
        print("batch inference results")
        print("="*50)
        
        valid_results = results_df.dropna()
        print(f"total WSIs processed: {len(results_df)}")
        print(f"successful predictions: {len(valid_results)}")
        
        if len(valid_results) > 0:
            print(f"\npredicted score statistics:")
            print(f"  mean: {valid_results['predicted_score'].mean():.3f}")
            print(f"  std:  {valid_results['predicted_score'].std():.3f}")
            print(f"  min:  {valid_results['predicted_score'].min():.3f}")
            print(f"  max:  {valid_results['predicted_score'].max():.3f}")
            
            print(f"\nscore distribution:")
            score_0_1 = ((valid_results['predicted_score'] >= 0) & (valid_results['predicted_score'] < 1)).sum()
            score_1_2 = ((valid_results['predicted_score'] >= 1) & (valid_results['predicted_score'] < 2)).sum()
            score_2_3 = ((valid_results['predicted_score'] >= 2) & (valid_results['predicted_score'] <= 3)).sum()
            
            print(f"  score 0-1: {score_0_1} WSIs")
            print(f"  score 1-2: {score_1_2} WSIs")
            print(f"  score 2-3: {score_2_3} WSIs")
            
            print(f"\nattention statistics:")
            print(f"  mean attention (avg): {valid_results['mean_attention'].mean():.8f}")
            print(f"  max attention (avg):  {valid_results['max_attention'].mean():.8f}")
        
        print(f"\ndetailed results saved to: {output_dir / 'batch_predictions.csv'}")
        
        if save_visualizations:
            print(f"individual visualizations saved to subdirectories in: {output_dir}")
            print(f"batch summary visualization: {output_dir / 'batch_summary.png'}")


if __name__ == "__main__":
    main() 