import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedKFold
import warnings
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
        # apply label smoohting for regression
        if self.label_smoothing == 0:
            return nn.MSELoss()(predictions, targets)
        
        # adding small noise to targets for regression
        noise = torch.randn_like(targets) * self.label_smoothing
        smoothed_targets = targets + noise
        
        # clamp to valid range [0, 3]
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
                
                # gradient clipping on backward pass
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


def train_ensemble_models(config_path: str, num_models: int = 5) -> List[Dict[str, Any]]:
    print(f"Training ensemble of {num_models} models...")
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    ensemble_results = []
    
    for model_idx in range(num_models):
        print(f"\nTraining Model {model_idx + 1}/{num_models}")
        
        # use different random seed for each model
        config = base_config.copy()
        config['random_seed'] = 42 + model_idx
        
        # Create model-specific checkpoint directory
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
    #Run stratified k-fold cross-validation
    print(f"Running {n_folds}-fold cross-validation...")
    
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
        
        # create fold-specific splits
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
        
        # create data loaders for this fold
        train_loader, val_loader, test_loader = create_data_loaders(
            config['data']['features_dir'], 
            config['data']['labels_file'], 
            fold_split_path, 
            config
        )
        
        # train model for this fold
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


def main():
    print("CLAM Regressor Trainig for Small Datasets")
    print("=" * 60)
    
    config_path = "configs/clam_regressor_training.yaml"
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    advanced_config = config.get('advanced', {})
    
    # fine-grained options
    if advanced_config.get('use_cross_validation', False):
        cv_folds = advanced_config.get('cv_folds', 5)
        run_cross_validation(config_path, cv_folds)

    elif advanced_config.get('ensemble_models', 0) > 1:
        num_models = advanced_config['ensemble_models']
        train_ensemble_models(config_path, num_models)    
    else:
        print("Single model training...")
        
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


if __name__ == "__main__":
    main() 