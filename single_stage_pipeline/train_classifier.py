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
import warnings
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from training.train_clam import CLAMTrainer
from training.dataset import create_data_loaders
from training.metrics import compute_metrics, format_metrics_for_logging, print_classification_report

warnings.filterwarnings("ignore")


class CLAMClassifierTrainer(CLAMTrainer):
    
    def __init__(self, config: Dict[str, Any], model_save_dir: str):
        
        self.config = config
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(config['hardware']['device'])
        print(f"Using device: {self.device}")
        
        self._set_random_seeds(config['random_seed'])
        
        from models.clam_classifier import create_clam_classifier
        self.model = create_clam_classifier(config).to(self.device)
        print(f"Created CLAM model with {self._count_parameters()} parameters")
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.use_amp = config['hardware']['mixed_precision']
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using automatic mixed precision")
        
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.train_history = []
        self.val_history = []
        
        self.patience = config['training']['early_stopping_patience']
        self.patience_counter = 0
        
        self.use_wandb = config['logging']['use_wandb']
        if self.use_wandb:
            self._init_wandb()
        
        self.gradient_clip_value = config.get('advanced', {}).get('gradient_clip_value', 1.0)
        self.class_names = config.get('classification', {}).get('class_names', ["T0", "T1", "T2", "T3"])
        self.binary_threshold = config.get('classification', {}).get('binary_threshold', 1.5)
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_bag_loss = 0.0
        total_instance_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} - Training")
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    results = self.model(features, labels)
                    
                    bag_loss = results['bag_loss']
                    instance_loss = results['instance_loss']
                    loss = results['loss']
                
                self.scaler.scale(loss).backward()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(features, labels)
                
                bag_loss = results['bag_loss']
                instance_loss = results['instance_loss']
                loss = results['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_bag_loss += bag_loss.item()
            total_instance_loss += instance_loss.item()
            
            with torch.no_grad():
                preds = self.model.predict(features)
                
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                if len(preds_np.shape) == 0:
                    preds_np = [preds_np.item()]
                    labels_np = [labels_np.item()]
                elif len(preds_np.shape) == 1:
                    preds_np = preds_np.tolist()
                    labels_np = labels_np.tolist()
                
                all_predictions.extend(preds_np)
                all_targets.extend(labels_np)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bag_loss': f"{bag_loss.item():.4f}",
                'inst_loss': f"{instance_loss.item():.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_bag_loss = total_bag_loss / len(train_loader)
        avg_instance_loss = total_instance_loss / len(train_loader)
        
        assert len(all_predictions) == len(all_targets), f"Prediction/target mismatch: {len(all_predictions)} vs {len(all_targets)}"
        
        metrics = compute_metrics(
            torch.tensor(all_predictions), 
            torch.tensor(all_targets),
            self.class_names,
            self.binary_threshold
        )
        
        result = {
            'loss': avg_loss,
            'bag_loss': avg_bag_loss,
            'instance_loss': avg_instance_loss,
        }
        
        result.update({f"train_{k}": v for k, v in format_metrics_for_logging(metrics).items()})
        
        return result
    
    def validate_epoch(self, val_loader) -> tuple:
        self.model.eval()
        
        total_loss = 0.0
        total_bag_loss = 0.0
        total_instance_loss = 0.0
        all_predictions = []
        all_targets = []
        slide_names = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} - Validation")
            
            for batch in pbar:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        results = self.model(features, labels)
                else:
                    results = self.model(features, labels)
                
                loss = results['loss']
                bag_loss = results['bag_loss']
                instance_loss = results['instance_loss']
                
                total_loss += loss.item()
                total_bag_loss += bag_loss.item()
                total_instance_loss += instance_loss.item()
                
                preds = self.model.predict(features)
                
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                if len(preds_np.shape) == 0:
                    preds_np = [preds_np.item()]
                    labels_np = [labels_np.item()]
                elif len(preds_np.shape) == 1:
                    preds_np = preds_np.tolist()
                    labels_np = labels_np.tolist()
                
                all_predictions.extend(preds_np)
                all_targets.extend(labels_np)
                slide_names.extend(batch['slide_names'])
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_bag_loss': f"{bag_loss.item():.4f}"
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_bag_loss = total_bag_loss / len(val_loader)
        avg_instance_loss = total_instance_loss / len(val_loader)
        
        assert len(all_predictions) == len(all_targets), f"Prediction/target mismatch: {len(all_predictions)} vs {len(all_targets)}"
        
        metrics = compute_metrics(
            torch.tensor(all_predictions), 
            torch.tensor(all_targets),
            self.class_names,
            self.binary_threshold
        )
        
        result = {
            'val_loss': avg_loss,
            'val_bag_loss': avg_bag_loss,
            'val_instance_loss': avg_instance_loss,
        }
        
        result.update({f"val_{k}": v for k, v in format_metrics_for_logging(metrics).items()})
        
        val_predictions = pd.DataFrame({
            'slide_name': slide_names,
            'prediction': all_predictions,
            'target': all_targets,
            'epoch': self.current_epoch
        })
        
        return result, val_predictions
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        print(f"\nStarting training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            train_results = self.train_epoch(train_loader)
            self.train_history.append(train_results)
            
            val_results, val_predictions = self.validate_epoch(val_loader)
            self.val_history.append(val_results)
            
            epoch_results = {**train_results, **val_results}
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['val_loss'])
                else:
                    self.scheduler.step()
            
            current_metric = self.get_metric_for_monitoring(val_results)
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self._save_checkpoint(epoch_results, is_best)
            
            if self.use_wandb:
                import wandb
                wandb.log(epoch_results)
            
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}:")
            print(f"  Train Loss: {train_results['loss']:.4f} | Val Loss: {val_results['val_loss']:.4f}")
            print(f"  Train Acc: {train_results.get('train_accuracy', 0):.4f} | Val Acc: {val_results.get('val_accuracy', 0):.4f}")
            print(f"  Train F1: {train_results.get('train_macro_f1', 0):.4f} | Val F1: {val_results.get('val_macro_f1', 0):.4f}")
            print(f"  Best Val Metric: {self.best_metric:.4f} | Patience: {self.patience_counter}/{self.patience}")
            
            pred_dir = self.model_save_dir / "predictions"
            pred_dir.mkdir(exist_ok=True)
            val_predictions.to_csv(pred_dir / f"val_predictions_epoch_{epoch+1}.csv", index=False)
            
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        history_df = pd.DataFrame(self.train_history + self.val_history)
        history_df.to_csv(self.model_save_dir / "training_history.csv", index=False)
        
        print(f"\nTraining completed! Best validation metric: {self.best_metric:.4f}")
        
        return {
            'best_metric': self.best_metric,
            'total_epochs': self.current_epoch + 1,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def _save_checkpoint(self, epoch_results: Dict[str, float], is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'epoch_results': epoch_results
        }
        
        checkpoint_path = self.model_save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.model_save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Val Metric: {self.best_metric:.4f}")

    def get_metric_for_monitoring(self, metrics: Dict[str, float]) -> float:
        monitor_metric = self.config['validation']['monitor']
        return metrics.get(monitor_metric, 0.0)


def create_holdout_split(labels_file: str, target_column: str, test_size: int = 18, 
                        random_state: int = 42) -> Tuple[List[str], List[str]]:
    # create held-out test set and return remaining slides for CV
    
    labels_df = pd.read_csv(labels_file)
    labels_df = labels_df.dropna(subset=[target_column])
    
    print(f"Total slides: {len(labels_df)}")
    print(f"Class distribution: {dict(labels_df[target_column].value_counts().sort_index())}")
    
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
    
    holdout_split = {
        'development': dev_slides,
        'test': test_slides
    }
    
    features_dir = config['data']['features_dir']
    splits_dir = Path(features_dir) / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    holdout_file = splits_dir / 'holdout_split.json'
    with open(holdout_file, 'w') as f:
        json.dump(holdout_split, f, indent=2)
    print(f"\nHoldout split saved to: {holdout_file}")
    
    labels_df = pd.read_csv(labels_file)
    labels_df = labels_df.dropna(subset=[target_column])
    dev_df = labels_df[labels_df['filename'].isin(dev_slides)]
    
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION ON DEVELOPMENT SET ({len(dev_slides)} slides)")
    print(f"{'='*50}")
    
    # run stratified CV on development set
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    dev_targets = dev_df[target_column].values
    
    cv_results = []
    all_predictions = []
    all_targets = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_targets)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        
        # get fold splits from devellopment set
        fold_train_slides = dev_df.iloc[train_idx]['filename'].tolist()
        fold_val_slides = dev_df.iloc[val_idx]['filename'].tolist()
        
        fold_splits = {
            'train': fold_train_slides,
            'val': fold_val_slides,
            'test': []  # no test in CV - we have held-out test
        }
        
        print(f"  Train: {len(fold_train_slides)} slides")
        print(f"  Val: {len(fold_val_slides)} slides")
        
        fold_split_file = splits_dir / f'cv_fold_{fold_idx}_splits.json'
        with open(fold_split_file, 'w') as f:
            json.dump(fold_splits, f, indent=2)
        
        train_loader, val_loader, _ = create_data_loaders(
            features_dir, labels_file, str(fold_split_file), config
        )
        
        fold_save_dir = f"{config['checkpoint']['save_dir']}/cv_fold_{fold_idx}"
        trainer = CLAMClassifierTrainer(config, fold_save_dir)
        training_results = trainer.train(train_loader, val_loader)
        
        val_metrics, val_predictions = trainer.validate_epoch(val_loader)
        
        fold_results = {
            'fold': fold_idx,
            'best_val_metric': training_results['best_metric'],
            'final_val_metrics': val_metrics,
            'train_size': len(fold_train_slides),
            'val_size': len(fold_val_slides)
        }
        
        cv_results.append(fold_results)
        
        all_predictions.extend(val_predictions['prediction'].tolist())
        all_targets.extend(val_predictions['target'].tolist())
        
        print(f"  Fold {fold_idx + 1} completed:")
        print(f"    Best Val Metric: {training_results['best_metric']:.4f}")
        print(f"    Final Val Acc: {val_metrics.get('val_accuracy', 0):.4f}")
        print(f"    Final Val F1: {val_metrics.get('val_macro_f1', 0):.4f}")
    
    overall_cv_metrics = compute_metrics(
        torch.tensor(all_predictions), 
        torch.tensor(all_targets),
        config.get('classification', {}).get('class_names', ["T0", "T1", "T2", "T3"]),
        config.get('classification', {}).get('binary_threshold', 1.5)
    )
    
    cv_summary = {
        'n_folds': n_folds,
        'development_size': len(dev_slides),
        'holdout_test_size': len(test_slides),
        'fold_results': cv_results,
        'cv_metrics': format_metrics_for_logging(overall_cv_metrics),
        'mean_val_metric': np.mean([r['best_val_metric'] for r in cv_results]),
        'std_val_metric': np.std([r['best_val_metric'] for r in cv_results])
    }
    
    cv_summary_path = Path(config['checkpoint']['save_dir']) / 'cv_summary.json'
    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS (Development Set)")
    print(f"{'='*60}")
    print(f"Mean Validation Metric: {cv_summary['mean_val_metric']:.4f} ± {cv_summary['std_val_metric']:.4f}")
    
    print_classification_report(
        np.array(all_targets), 
        np.array(all_predictions),
        config.get('classification', {}).get('class_names', ["T0", "T1", "T2", "T3"])
    )
    
    return cv_summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CLAM Classifier with Held-out Test Set')
    parser.add_argument('--config', type=str, 
                        default='configs/clam_classifier_training.yaml',
                        help='Path to config file')
    parser.add_argument('--test-size', type=int, default=18,
                        help='Number of slides to hold out for testing')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    cv_results = run_cross_validation_with_holdout(
        args.config, n_folds=args.cv_folds, test_size=args.test_size
    )
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("1. Use best performing fold model OR")
    print("2. Train ensemble on full development set")
    print("3. Evaluate on held-out test set using evaluate_classifier.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 