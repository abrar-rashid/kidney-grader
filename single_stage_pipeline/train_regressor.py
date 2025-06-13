
# run from the kidney-grader root directory:
# python single_stage_pipeline/train_regressor.py --config single_stage_pipeline/configs/clam_regressor.yaml --mode single
# python single_stage_pipeline/train_regressor.py --config single_stage_pipeline/configs/clam_regressor.yaml --mode cv --n_folds 5

# run from the single_stage_pipeline directory:
# python train_regressor.py --config configs/clam_regressor.yaml --mode single
# python train_regressor.py --config configs/clam_regressor.yaml --mode cv --n_folds 5

# train ensemble models:
# python single_stage_pipeline/train_regressor.py --config single_stage_pipeline/configs/clam_regressor.yaml --mode ensemble --num_models 5


import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from training.train_clam import CLAMTrainer
from training.dataset import create_data_loaders

warnings.filterwarnings("ignore")


class CLAMRegressorTrainer(CLAMTrainer):
    
    def __init__(self, config: Dict[str, Any], model_save_dir: str):
        super().__init__(config, model_save_dir)
        
        self.gradient_clip_value = config.get('advanced', {}).get('gradient_clip_value', 1.0)
        self.label_smoothing = config.get('advanced', {}).get('label_smoothing', 0.1)
    
    def _label_smoothing_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing == 0:
            return nn.MSELoss()(predictions, targets)
        
        noise = torch.randn_like(targets) * self.label_smoothing
        smoothed_targets = targets + noise
        
        smoothed_targets = torch.clamp(smoothed_targets, 0.0, 3.0)
        
        return nn.MSELoss()(predictions, smoothed_targets)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_bag_loss = 0.0
        total_instance_loss = 0.0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} - Training")
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    results = self.model(features, labels)
                    
                    bag_loss = self._label_smoothing_loss(results['logits'].squeeze(), labels.float())
                    instance_loss = results['instance_loss']
                    loss = self.config['training']['bag_loss_weight'] * bag_loss + \
                           self.config['training']['instance_loss_weight'] * instance_loss
                
                self.scaler.scale(loss).backward()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(features, labels)
                
                bag_loss = self._label_smoothing_loss(results['logits'].squeeze(), labels.float())
                instance_loss = results['instance_loss']
                loss = self.config['training']['bag_loss_weight'] * bag_loss + \
                       self.config['training']['instance_loss_weight'] * instance_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_bag_loss += bag_loss.item()
            total_instance_loss += instance_loss.item()
            
            preds = results['logits'].squeeze().detach().cpu().numpy()
            targs = labels.detach().cpu().numpy()
            
            if len(preds.shape) == 0:
                preds = [preds.item()]
                targs = [targs.item()]
            
            predictions.extend(preds)
            targets.extend(targs)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bag_loss': f"{bag_loss.item():.4f}",
                'inst_loss': f"{instance_loss.item():.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_bag_loss = total_bag_loss / len(train_loader)
        avg_instance_loss = total_instance_loss / len(train_loader)
        
        metrics = self.calculate_metrics(np.array(predictions), np.array(targets))
        
        return {
            'loss': avg_loss,
            'bag_loss': avg_bag_loss,
            'instance_loss': avg_instance_loss,
            **{f"train_{k}": v for k, v in metrics.items()}
        }


def create_holdout_split(labels_file: str, target_column: str, test_size: int = 18, 
                        random_state: int = 42) -> Tuple[List[str], List[str]]:
    
    labels_df = pd.read_csv(labels_file)
    labels_df = labels_df.dropna(subset=[target_column])
    
    print(f"Total slides: {len(labels_df)}")
    print(f"Class distribution: {dict(labels_df[target_column].value_counts().sort_index())}")
    
    # stratified split to hold out test set
    slides = labels_df['filename'].tolist()
    targets = labels_df[target_column].values
    
    dev_slides, test_slides, dev_targets, test_targets = train_test_split(
        slides, targets, 
        test_size=test_size,
        stratify=targets,
        random_state=random_state
    )
    
    print(f"\nHeld-out split:")
    print(f"  Development set: {len(dev_slides)} slides")
    print(f"  Test set: {len(test_slides)} slides")
    
    dev_dist = dict(zip(*np.unique(dev_targets, return_counts=True)))
    test_dist = dict(zip(*np.unique(test_targets, return_counts=True)))
    
    print(f"  Dev class distribution: {dev_dist}")
    print(f"  Test class distribution: {test_dist}")
    
    return dev_slides, test_slides


def train_single_model(config_path: str) -> Dict[str, Any]:
    print("="*60)
    print("SINGLE MODEL TRAINING")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    features_dir = config['data']['features_dir']
    labels_file = config['data']['labels_file']
    split_file = f"{features_dir}/splits/data_splits.json"
    
    train_loader, val_loader, test_loader = create_data_loaders(
        features_dir, labels_file, split_file, config
    )
    
    model_save_dir = config['checkpoint']['save_dir']
    trainer = CLAMRegressorTrainer(config, model_save_dir)
    training_results = trainer.train(train_loader, val_loader)
    
    print(f"Training completed - Best Val MSE: {training_results['best_metric']:.4f}")
    
    return training_results


def train_ensemble_models(config_path: str, num_models: int = 5) -> List[Dict[str, Any]]:
    print("="*60)
    print(f"ENSEMBLE TRAINING ({num_models} models)")
    print("="*60)
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    ensemble_results = []
    
    for model_idx in range(num_models):
        print(f"\nTraining Model {model_idx + 1}/{num_models}")
        
        config = base_config.copy()
        config['random_seed'] = 42 + model_idx
        
        model_save_dir = f"{base_config['checkpoint']['save_dir']}/model_{model_idx}"
        
        features_dir = config['data']['features_dir']
        labels_file = config['data']['labels_file']
        split_file = f"{features_dir}/splits/data_splits.json"
        
        train_loader, val_loader, test_loader = create_data_loaders(
            features_dir, labels_file, split_file, config
        )
        
        trainer = CLAMRegressorTrainer(config, model_save_dir)
        training_results = trainer.train(train_loader, val_loader)
        
        ensemble_results.append({
            'model_idx': model_idx,
            'best_val_mse': training_results['best_metric'],
            'total_epochs': training_results['total_epochs'],
            'model_path': f"{model_save_dir}/best_model.pth"
        })
        
        print(f"Model {model_idx + 1} completed - Best Val MSE: {training_results['best_metric']:.4f}")
    
    ensemble_summary = {
        'ensemble_size': num_models,
        'models': ensemble_results,
        'mean_val_mse': np.mean([r['best_val_mse'] for r in ensemble_results]),
        'std_val_mse': np.std([r['best_val_mse'] for r in ensemble_results])
    }
    
    summary_path = Path(base_config['checkpoint']['save_dir']) / 'ensemble_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(ensemble_summary, f, indent=2)
    
    print(f"\nEnsemble training completed!")
    print(f"Mean validation MSE: {ensemble_summary['mean_val_mse']:.4f} ± {ensemble_summary['std_val_mse']:.4f}")
    
    return ensemble_results


def run_cross_validation(config_path: str, n_folds: int = 5) -> Dict[str, Any]:
    # run stratified k-fold cross-validation using all data
    print("="*60)
    print(f"CROSS-VALIDATION ({n_folds} folds, all data)")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # load all data for CV splits
    labels_df = pd.read_csv(config['data']['labels_file'])
    labels_df = labels_df.dropna(subset=[config['data']['target_column']])
    
    # stratified k-fold based on tubulitis scores
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    targets = labels_df[config['data']['target_column']].values
    
    cv_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labels_df, targets)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        
        train_slides = labels_df.iloc[train_idx]['filename'].tolist()
        val_slides = labels_df.iloc[val_idx]['filename'].tolist()
        
        fold_splits = {
            'train': train_slides,
            'val': val_slides,
            'test': val_slides  # use validation as test for CV
        }
        
        fold_split_path = f"data/features/splits/cv_fold_{fold_idx}_splits.json"
        os.makedirs(os.path.dirname(fold_split_path), exist_ok=True)
        with open(fold_split_path, 'w') as f:
            json.dump(fold_splits, f, indent=2)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            config['data']['features_dir'], 
            config['data']['labels_file'], 
            fold_split_path, 
            config
        )
        
        fold_save_dir = f"{config['checkpoint']['save_dir']}/cv_fold_{fold_idx}"
        trainer = CLAMRegressorTrainer(config, fold_save_dir)
        fold_results = trainer.train(train_loader, val_loader)
        
        cv_results.append({
            'fold': fold_idx,
            'best_val_mse': fold_results['best_metric'],
            'total_epochs': fold_results['total_epochs']
        })
        
        print(f"Fold {fold_idx + 1} completed - Val MSE: {fold_results['best_metric']:.4f}")
    
    val_mses = [r['best_val_mse'] for r in cv_results]
    cv_summary = {
        'cv_folds': n_folds,
        'fold_results': cv_results,
        'mean_val_mse': np.mean(val_mses),
        'std_val_mse': np.std(val_mses),
        'min_val_mse': np.min(val_mses),
        'max_val_mse': np.max(val_mses)
    }
    
    cv_path = Path(config['checkpoint']['save_dir']) / 'cross_validation_results.json'
    with open(cv_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\nCross-validation completed!")
    print(f"CV MSE: {cv_summary['mean_val_mse']:.4f} ± {cv_summary['std_val_mse']:.4f}")
    print(f"Range: [{cv_summary['min_val_mse']:.4f}, {cv_summary['max_val_mse']:.4f}]")
    
    return cv_summary


def run_cross_validation_with_holdout(config_path: str, n_folds: int = 5, 
                                     test_size: int = 18) -> Dict[str, Any]:
    
    print("="*80)
    print("HELD-OUT TEST SET + CROSS-VALIDATION")
    print("="*80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    labels_file = config['data']['labels_file']
    target_column = config['data']['target_column']
    
    dev_slides, test_slides = create_holdout_split(
        labels_file, target_column, test_size=test_size, random_state=42
    )
    
    # save holdout split for later evaluation
    holdout_split = {
        'development': dev_slides,
        'test': test_slides
    }
    
    features_dir = config['data']['features_dir']
    splits_dir = Path(features_dir) / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    holdout_file = splits_dir / 'holdout_split_regressor.json'
    with open(holdout_file, 'w') as f:
        json.dump(holdout_split, f, indent=2)
    print(f"\nHoldout split saved to: {holdout_file}")
    
    labels_df = pd.read_csv(labels_file)
    labels_df = labels_df.dropna(subset=[target_column])
    dev_df = labels_df[labels_df['filename'].isin(dev_slides)]
    
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION ON DEVELOPMENT SET ({len(dev_slides)} slides)")
    print(f"{'='*50}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    dev_targets = dev_df[target_column].values
    
    cv_results = []
    all_predictions = []
    all_targets = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_targets)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        
        fold_train_slides = dev_df.iloc[train_idx]['filename'].tolist()
        fold_val_slides = dev_df.iloc[val_idx]['filename'].tolist()
        
        fold_splits = {
            'train': fold_train_slides,
            'val': fold_val_slides,
            'test': []  # no test in CV (held-out test)
        }
        
        print(f"  Train: {len(fold_train_slides)} slides")
        print(f"  Val: {len(fold_val_slides)} slides")
        
        fold_split_file = splits_dir / f'cv_fold_{fold_idx}_regressor_splits.json'
        with open(fold_split_file, 'w') as f:
            json.dump(fold_splits, f, indent=2)
        
        train_loader, val_loader, _ = create_data_loaders(
            features_dir, labels_file, str(fold_split_file), config
        )
        
        fold_save_dir = f"{config['checkpoint']['save_dir']}/cv_fold_{fold_idx}"
        trainer = CLAMRegressorTrainer(config, fold_save_dir)
        training_results = trainer.train(train_loader, val_loader)
        
        val_results, val_predictions = trainer.validate_epoch(val_loader)
        
        val_preds = val_predictions['prediction'].values
        val_targets = val_predictions['target'].values
        
        mse = mean_squared_error(val_targets, val_preds)
        mae = mean_absolute_error(val_targets, val_preds)
        rmse = np.sqrt(mse)
        correlation, _ = pearsonr(val_targets, val_preds)
        
        fold_results = {
            'fold': fold_idx,
            'best_val_mse': training_results['best_metric'],
            'final_val_mse': val_results['val_mse'],
            'final_val_mae': mae,
            'final_val_rmse': rmse,
            'final_val_correlation': correlation,
            'train_size': len(fold_train_slides),
            'val_size': len(fold_val_slides)
        }
        
        cv_results.append(fold_results)
        
        all_predictions.extend(val_preds.tolist())
        all_targets.extend(val_targets.tolist())
        
        print(f"  Fold {fold_idx + 1} completed:")
        print(f"    Best Val MSE: {training_results['best_metric']:.4f}")
        print(f"    Final Val MSE: {mse:.4f}")
        print(f"    Final Val MAE: {mae:.4f}")
        print(f"    Final Val Correlation: {correlation:.4f}")
    
    overall_mse = mean_squared_error(all_targets, all_predictions)
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_rmse = np.sqrt(overall_mse)
    overall_correlation, _ = pearsonr(all_targets, all_predictions)
    
    cv_summary = {
        'n_folds': n_folds,
        'development_size': len(dev_slides),
        'holdout_test_size': len(test_slides),
        'fold_results': cv_results,
        'overall_cv_metrics': {
            'mse': overall_mse,
            'mae': overall_mae,
            'rmse': overall_rmse,
            'correlation': overall_correlation
        },
        'mean_val_mse': np.mean([r['best_val_mse'] for r in cv_results]),
        'std_val_mse': np.std([r['best_val_mse'] for r in cv_results]),
        'mean_final_mse': np.mean([r['final_val_mse'] for r in cv_results]),
        'mean_final_mae': np.mean([r['final_val_mae'] for r in cv_results]),
        'mean_final_correlation': np.mean([r['final_val_correlation'] for r in cv_results])
    }
    
    cv_summary_path = Path(config['checkpoint']['save_dir']) / 'cv_summary_regressor.json'
    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS (Development Set)")
    print(f"{'='*60}")
    print(f"Mean Validation MSE: {cv_summary['mean_val_mse']:.4f} ± {cv_summary['std_val_mse']:.4f}")
    print(f"Overall CV Metrics:")
    print(f"  MSE: {overall_mse:.4f}")
    print(f"  MAE: {overall_mae:.4f}")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  Correlation: {overall_correlation:.4f}")
    
    print(f"\nRegression Analysis:")
    print(f"  Target range: [{min(all_targets):.1f}, {max(all_targets):.1f}]")
    print(f"  Prediction range: [{min(all_predictions):.2f}, {max(all_predictions):.2f}]")
    
    rounded_preds = np.round(all_predictions).astype(int)
    rounded_targets = np.round(all_targets).astype(int)
    
    rounded_preds = np.clip(rounded_preds, 0, 3)
    rounded_targets = np.clip(rounded_targets, 0, 3)
    
    exact_accuracy = np.mean(rounded_preds == rounded_targets)
    adjacent_accuracy = np.mean(np.abs(rounded_preds - rounded_targets) <= 1)
    
    print(f"  Rounded Accuracy: {exact_accuracy:.4f}")
    print(f"  Adjacent Accuracy (±1): {adjacent_accuracy:.4f}")
    
    return cv_summary


def main():
    parser = argparse.ArgumentParser(description='Train CLAM Regressor Models')
    
    parser.add_argument('--mode', choices=['single', 'ensemble', 'cv', 'cv-holdout'], 
                       default='cv-holdout',
                       help='Training mode (default: cv-holdout)')
    
    parser.add_argument('--config', type=str, 
                       default='configs/clam_regressor_training.yaml',
                       help='Path to config file')
    
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--ensemble-size', type=int, default=5,
                       help='Number of models for ensemble training')
    parser.add_argument('--holdout-size', type=int, default=18,
                       help='Number of slides to hold out for testing (cv-holdout mode)')
    
    args = parser.parse_args()
    
    print("CLAM Regressor Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    
    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        return
    
    if args.mode == 'single':
        results = train_single_model(args.config)
        
    elif args.mode == 'ensemble':
        results = train_ensemble_models(args.config, args.ensemble_size)
        
    elif args.mode == 'cv':
        results = run_cross_validation(args.config, args.cv_folds)
        
    elif args.mode == 'cv-holdout':
        results = run_cross_validation_with_holdout(
            args.config, args.cv_folds, args.holdout_size
        )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    
    if args.mode == 'cv-holdout':
        print("\nNEXT STEPS:")
        print("1. Use best performing fold model OR")
        print("2. Train ensemble on full development set")
        print("3. Evaluate on held-out test set using:")
        print("   python evaluate_regressor.py --holdout --model-path <best_model>")
        print("   python evaluate_regressor.py --cv-ensemble --checkpoints-dir <cv_dir>")
    else:
        print("\nNEXT STEPS:")
        print("1. Evaluate trained models using:")
        print("   python evaluate_regressor.py --all-folds")
        print("   python evaluate_regressor.py --fold <specific_fold>")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 