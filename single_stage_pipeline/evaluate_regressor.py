import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from models import create_clam_regressor
from training import create_data_loaders, CLAMTrainer

warnings.filterwarnings("ignore")


class CLAMRegressorEvaluator:    
    def __init__(self, config_path: str, model_path: str):
        self.config_path = config_path
        self.model_path = model_path
        
        #load config yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        self.model = create_clam_regressor(self.config).to(self.device)        
        self.load_model()
        
        print(f"Loaded model from: {model_path}")
        print(f"Model parameters: {self.count_parameters():,}")
    
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Best validation MSE: {checkpoint.get('best_metric', 'unknown'):.4f}")
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded model weights")
        
        self.model.eval()
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def evaluate_dataset(self, data_loader, split_name: str) -> Dict[str, Any]:
        print(f"\nEvaluating on {split_name} set...")
        
        predictions = []
        targets = []
        slide_names = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                results = self.model(features)
                
                preds = results['logits'].squeeze().cpu().numpy()
                targs = labels.cpu().numpy()
                
                if len(preds.shape) == 0:
                    preds = [preds.item()]
                    targs = [targs.item()]
                
                predictions.extend(preds)
                targets.extend(targs)
                slide_names.extend(batch['slide_names'])
                
                # store attention weights for analysis
                attn_weights = results['attention_weights'].squeeze().cpu().numpy()
                attention_weights_list.append(attn_weights)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        results_df = pd.DataFrame({
            'slide_name': slide_names,
            'prediction': predictions,
            'target': targets,
            'error': predictions - targets,
            'abs_error': np.abs(predictions - targets)
        })
        
        return {
            'metrics': metrics,
            'predictions': results_df,
            'attention_weights': attention_weights_list,
            'split_name': split_name
        }
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # correlation metrics
        pearson_corr, pearson_p = pearsonr(predictions, targets)
        spearman_corr, spearman_p = spearmanr(predictions, targets)
        
        # classification-style metrics (exact match)
        rounded_preds = np.round(predictions).astype(int)
        rounded_targets = np.round(targets).astype(int)
        exact_accuracy = np.mean(rounded_preds == rounded_targets)
        
        # within-1 accuracy (predictions within 1 score of target)
        within_1_accuracy = np.mean(np.abs(predictions - targets) <= 1.0)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'exact_accuracy': exact_accuracy,
            'within_1_accuracy': within_1_accuracy
        }
    
    def plot_results(self, results: Dict[str, Any], save_dir: str):
        predictions_df = results['predictions']
        split_name = results['split_name']
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(predictions_df['target'], predictions_df['prediction'], 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        min_val = min(predictions_df['target'].min(), predictions_df['prediction'].min())
        max_val = max(predictions_df['target'].max(), predictions_df['prediction'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Ground Truth Tubulitis Score', fontsize=12)
        plt.ylabel('Predicted Tubulitis Score', fontsize=12)
        plt.title(f'CLAM Regressor: Predictions vs Ground Truth ({split_name})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        metrics = results['metrics']
        metrics_text = f"R² = {metrics['r2']:.3f}\nPearson r = {metrics['pearson_correlation']:.3f}\nMAE = {metrics['mae']:.3f}\nRMSE = {metrics['rmse']:.3f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path / f'{split_name}_predictions_vs_targets.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(predictions_df['error'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Prediction Error (Predicted - Target)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Error Distribution ({split_name})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / f'{split_name}_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        predictions_df['target_rounded'] = predictions_df['target'].round().astype(int)
        sns.boxplot(data=predictions_df, x='target_rounded', y='prediction')
        plt.xlabel('Ground Truth T Score', fontsize=12)
        plt.ylabel('Predicted Score', fontsize=12)
        plt.title(f'Prediction Distribution by Ground Truth Class ({split_name})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / f'{split_name}_predictions_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        rounded_preds = predictions_df['prediction'].round().astype(int)
        rounded_targets = predictions_df['target_rounded']
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(rounded_targets, rounded_preds, labels=[0, 1, 2, 3])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['T0', 'T1', 'T2', 'T3'],
                   yticklabels=['T0', 'T1', 'T2', 'T3'])
        plt.xlabel('Predicted T Score', fontsize=12)
        plt.ylabel('Ground Truth T Score', fontsize=12)
        plt.title(f'Confusion Matrix (Rounded Predictions) ({split_name})', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path / f'{split_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {save_path}")
    
    def evaluate_all_splits(self, save_dir: str) -> Dict[str, Any]:
        features_dir = self.config['data']['features_dir']
        labels_file = self.config['data']['labels_file']
        split_file = f"{features_dir}/splits/data_splits.json"
        
        train_loader, val_loader, test_loader = create_data_loaders(
            features_dir, labels_file, split_file, self.config
        )
        
        all_results = {}
        
        for loader, split_name in [(train_loader, 'train'), (val_loader, 'val'), (test_loader, 'test')]:
            results = self.evaluate_dataset(loader, split_name)
            all_results[split_name] = results
            
            metrics = results['metrics']
            print(f"\n{split_name.upper()} METRICS:")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Pearson r: {metrics['pearson_correlation']:.4f}")
            print(f"  Exact Accuracy: {metrics['exact_accuracy']:.1%}")
            print(f"  Within-1 Accuracy: {metrics['within_1_accuracy']:.1%}")
            
            self.plot_results(results, f"{save_dir}/{split_name}")
        
        self.create_summary_comparison(all_results, save_dir)
        
        return all_results
    
    def create_summary_comparison(self, all_results: Dict[str, Any], save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        metrics_df = pd.DataFrame({
            split: results['metrics'] 
            for split, results in all_results.items()
        }).T
        
        print(f"\nSUMMARY COMPARISON:")
        print(metrics_df.round(4))
        
        metrics_df.to_csv(save_path / 'metrics_comparison.csv')
        
        plt.figure(figsize=(15, 5))
        
        for i, (split, results) in enumerate(all_results.items()):
            plt.subplot(1, 3, i+1)
            df = results['predictions']
            
            plt.scatter(df['target'], df['prediction'], alpha=0.6, s=30)
            min_val = min(df['target'].min(), df['prediction'].min())
            max_val = max(df['target'].max(), df['prediction'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            plt.xlabel('Ground Truth')
            plt.ylabel('Predicted')
            plt.title(f'{split.upper()}\nR² = {results["metrics"]["r2"]:.3f}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics_df


def create_test_only_loader(
    features_dir: str,
    labels_file: str,
    split_file: str,
    config: Dict[str, Any]
) -> torch.utils.data.DataLoader:
    from training.dataset import WSIFeaturesDataset, collate_fn
    
    data_config = config['data']
    
    test_dataset = WSIFeaturesDataset(
        features_dir=features_dir,
        labels_file=labels_file,
        split_file=split_file,
        split_name='test',
        target_column=data_config['target_column'],
        max_patches=None,
        augment=False
    )
    
    print(f"Test dataset: {len(test_dataset)} slides")
    print(f"Test label distribution: {test_dataset.get_label_distribution()}")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return test_loader


def evaluate_regressor_on_holdout_test(model_path: str, config_path: str, output_dir: str = None) -> Dict[str, Any]:
    print("="*60)
    print("EVALUATING REGRESSOR ON HELD-OUT TEST SET")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # load holdout split
    features_dir = config['data']['features_dir']
    holdout_file = Path(features_dir) / 'splits' / 'holdout_split_regressor.json'
    
    if not holdout_file.exists():
        raise FileNotFoundError(f"Holdout split file not found: {holdout_file}")
    
    with open(holdout_file, 'r') as f:
        holdout_split = json.load(f)
    
    test_slides = holdout_split['test']
    print(f"Evaluating on {len(test_slides)} held-out test slides")
    
    # create test data loader
    test_split = {
        'train': [],
        'val': [],
        'test': test_slides
    }
    
    temp_split_file = Path(features_dir) / 'splits' / 'temp_holdout_test_regressor.json'
    with open(temp_split_file, 'w') as f:
        json.dump(test_split, f, indent=2)
    
    try:
        test_loader = create_test_only_loader(
            features_dir, config['data']['labels_file'], str(temp_split_file), config
        )

        device = torch.device(config['hardware']['device'])
        model = create_clam_regressor(config).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Model trained for {checkpoint['epoch']} epochs")
        
        all_predictions = []
        all_targets = []
        slide_names = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                results = model(features)
                preds = results['logits'].squeeze()
                
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
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        
        pearson_corr, pearson_p = pearsonr(all_predictions, all_targets)
        spearman_corr, spearman_p = spearmanr(all_predictions, all_targets)
        
        # round regressions to nearest integer for classification-style metrics
        rounded_preds = np.round(all_predictions).astype(int)
        rounded_targets = np.round(all_targets).astype(int)
        
        rounded_preds = np.clip(rounded_preds, 0, 3)
        rounded_targets = np.clip(rounded_targets, 0, 3)
        
        exact_accuracy = np.mean(rounded_preds == rounded_targets)
        adjacent_accuracy = np.mean(np.abs(rounded_preds - rounded_targets) <= 1)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'exact_accuracy': exact_accuracy,
            'adjacent_accuracy': adjacent_accuracy
        }
        
        results_df = pd.DataFrame({
            'slide_name': slide_names,
            'prediction': all_predictions,
            'target': all_targets,
            'error': all_predictions - all_targets,
            'abs_error': np.abs(all_predictions - all_targets)
        })
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_df.to_csv(output_path / 'holdout_test_results_regressor.csv', index=False)
            
            with open(output_path / 'holdout_test_metrics_regressor.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Results saved to: {output_path}")
        
        print(f"\n{'='*50}")
        print("HELD-OUT TEST SET RESULTS (REGRESSOR)")
        print(f"{'='*50}")
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nRegression Analysis:")
        print(f"  Target range: [{all_targets.min():.1f}, {all_targets.max():.1f}]")
        print(f"  Prediction range: [{all_predictions.min():.2f}, {all_predictions.max():.2f}]")
        
        return {
            'metrics': metrics,
            'predictions': results_df,
            'num_samples': len(all_predictions)
        }
        
    finally:
        if temp_split_file.exists():
            temp_split_file.unlink()


def evaluate_regressor_cv_ensemble_on_holdout(cv_dir: str, config_path: str, output_dir: str = None) -> Dict[str, Any]:
    
    print("="*70)
    print("EVALUATING REGRESSOR CV ENSEMBLE ON HELD-OUT TEST SET")
    print("="*70)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cv_path = Path(cv_dir)
    fold_models = []
    
    for fold_dir in cv_path.glob("cv_fold_*"):
        best_model = fold_dir / "best_model.pth"
        if best_model.exists():
            fold_models.append(str(best_model))
    
    if not fold_models:
        raise FileNotFoundError(f"No CV fold models found in {cv_dir}")
    
    print(f"Found {len(fold_models)} CV fold models")
    
    features_dir = config['data']['features_dir']
    holdout_file = Path(features_dir) / 'splits' / 'holdout_split_regressor.json'
    
    with open(holdout_file, 'r') as f:
        holdout_split = json.load(f)
    
    test_slides = holdout_split['test']
    print(f"Evaluating on {len(test_slides)} held-out test slides")

    test_split = {
        'train': [],
        'val': [],
        'test': test_slides
    }
    
    temp_split_file = Path(features_dir) / 'splits' / 'temp_holdout_test_regressor.json'
    with open(temp_split_file, 'w') as f:
        json.dump(test_split, f, indent=2)
    
    try:
        test_loader = create_test_only_loader(
            features_dir, config['data']['labels_file'], str(temp_split_file), config
        )
        
        device = torch.device(config['hardware']['device'])
        models = []
        
        for model_path in fold_models:
            model = create_clam_regressor(config).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        
        print(f"Loaded {len(models)} models for ensemble")
        
        all_ensemble_predictions = []
        all_targets = []
        slide_names = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Ensemble Evaluation"):
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                batch_preds = []
                for model in models:
                    results = model(features)
                    preds = results['logits'].squeeze()
                    batch_preds.append(preds.cpu().numpy())
                
                # average predictions
                ensemble_preds = np.mean(batch_preds, axis=0)
                
                labels_np = labels.cpu().numpy()
                
                if len(ensemble_preds.shape) == 0:
                    ensemble_preds = [ensemble_preds.item()]
                    labels_np = [labels_np.item()]
                elif len(ensemble_preds.shape) == 1:
                    ensemble_preds = ensemble_preds.tolist()
                    labels_np = labels_np.tolist()
                
                all_ensemble_predictions.extend(ensemble_preds)
                all_targets.extend(labels_np)
                slide_names.extend(batch['slide_names'])
        
        all_ensemble_predictions = np.array(all_ensemble_predictions)
        all_targets = np.array(all_targets)

        mse = mean_squared_error(all_targets, all_ensemble_predictions)
        mae = mean_absolute_error(all_targets, all_ensemble_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_ensemble_predictions)
        
        pearson_corr, pearson_p = pearsonr(all_ensemble_predictions, all_targets)
        spearman_corr, spearman_p = spearmanr(all_ensemble_predictions, all_targets)
        
        rounded_preds = np.round(all_ensemble_predictions).astype(int)
        rounded_targets = np.round(all_targets).astype(int)
        
        rounded_preds = np.clip(rounded_preds, 0, 3)
        rounded_targets = np.clip(rounded_targets, 0, 3)
        
        exact_accuracy = np.mean(rounded_preds == rounded_targets)
        adjacent_accuracy = np.mean(np.abs(rounded_preds - rounded_targets) <= 1)
        
        ensemble_metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'exact_accuracy': exact_accuracy,
            'adjacent_accuracy': adjacent_accuracy
        }
        
        ensemble_results_df = pd.DataFrame({
            'slide_name': slide_names,
            'ensemble_prediction': all_ensemble_predictions,
            'target': all_targets,
            'error': all_ensemble_predictions - all_targets,
            'abs_error': np.abs(all_ensemble_predictions - all_targets)
        })
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            ensemble_results_df.to_csv(output_path / 'holdout_ensemble_results_regressor.csv', index=False)
            
            with open(output_path / 'holdout_ensemble_metrics_regressor.json', 'w') as f:
                json.dump(ensemble_metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print("ENSEMBLE HELD-OUT TEST SET RESULTS (REGRESSOR)")
        print(f"{'='*60}")
        
        for metric, value in ensemble_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nRegression Analysis:")
        print(f"  Target range: [{all_targets.min():.1f}, {all_targets.max():.1f}]")
        print(f"  Prediction range: [{all_ensemble_predictions.min():.2f}, {all_ensemble_predictions.max():.2f}]")
        
        return {
            'ensemble_metrics': ensemble_metrics,
            'ensemble_predictions': ensemble_results_df,
            'num_models': len(models),
            'num_samples': len(all_ensemble_predictions)
        }
        
    finally:
        if temp_split_file.exists():
            temp_split_file.unlink()

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLAM regressor models")
    parser.add_argument("--fold", type=str, help="Specific fold to evaluate (e.g., cv_fold_0)")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate all available folds")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints_regressor", help="Directory containing model checkpoints (default: checkpoints_regressor)")
    parser.add_argument("--config", type=str, default="configs/clam_regressor_training.yaml", help="Path to config file")
    parser.add_argument("--holdout", action="store_true", help="Evaluate single model on held-out test set")
    parser.add_argument("--cv-ensemble", action="store_true", help="Evaluate CV ensemble on held-out test set")
    parser.add_argument("--model-path", type=str, help="Path to specific model for holdout evaluation")
    parser.add_argument("--output-dir", type=str, help="Directory to save evaluation results")
    args = parser.parse_args()
    
    print("CLAM Regressor Model Evaluation")
    print("=" * 50)
    
    if args.holdout:
        if not args.model_path:
            raise ValueError("--model-path required for held-out evaluation")
        evaluate_regressor_on_holdout_test(args.model_path, args.config, args.output_dir)
        return
    
    elif args.cv_ensemble:
        if not args.checkpoints_dir:
            raise ValueError("--checkpoints-dir required for CV ensemble evaluation")
        evaluate_regressor_cv_ensemble_on_holdout(args.checkpoints_dir, args.config, args.output_dir)
        return
    
    config_path = args.config
    checkpoints_dir = Path(args.checkpoints_dir)
    
    if not checkpoints_dir.exists():
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        print("Please run regressor training first!")
        return
    
    cv_folds = list(checkpoints_dir.glob("cv_fold_*"))
    
    if cv_folds:
        print(f"Found {len(cv_folds)} cross-validation folds")
        
        fold_models = []
        for fold_dir in sorted(cv_folds):
            best_model_path = fold_dir / "best_model.pth"
            if best_model_path.exists():
                fold_models.append({
                    'fold_name': fold_dir.name,
                    'model_path': best_model_path,
                    'fold_dir': fold_dir
                })
                print(f"{fold_dir.name}: {best_model_path}")
            else:
                print(f"{fold_dir.name}: No best_model.pth found")
        
        if not fold_models:
            print("No trained models found in any fold!")
            return
        
        if args.fold:
            # evaluate specific fold
            selected_folds = [f for f in fold_models if f['fold_name'] == args.fold]
            if not selected_folds:
                print(f"Fold '{args.fold}' not found!")
                print("Available folds:", [f['fold_name'] for f in fold_models])
                return
            folds_to_evaluate = selected_folds
        elif args.all_folds:
            folds_to_evaluate = fold_models
        else:
            # default is evaluate first fold only
            folds_to_evaluate = [fold_models[0]]
            print(f"\n🏆 Evaluating fold: {fold_models[0]['fold_name']} (use --all-folds to evaluate all)")
        
        for selected_fold in folds_to_evaluate:
            print(f"\n{'='*50}")
            print(f"Evaluating fold: {selected_fold['fold_name']}")
            print(f"{'='*50}")
            
            best_model_path = selected_fold['model_path']
            
            evaluator = CLAMRegressorEvaluator(config_path, best_model_path)
            
            save_dir = f"evaluation_results_regressor_{selected_fold['fold_name']}"
            all_results = evaluator.evaluate_all_splits(save_dir)
            
            print(f"\nEvaluation completed for {selected_fold['fold_name']}!")
            print(f"Results saved to: {save_dir}")
            
            save_path = Path(save_dir)
            for split, results in all_results.items():
                results['predictions'].to_csv(save_path / f'{split}_detailed_predictions.csv', index=False)
        
        if len(folds_to_evaluate) == 1 and not args.fold and not args.all_folds:
            print(f"\nTo evaluate other folds:")
            print(f"   python evaluate_regressor.py --all-folds  # Evaluate all folds")
            for fold in fold_models:
                print(f"   python evaluate_regressor.py --fold {fold['fold_name']}")
            print(f"\nTo evaluate on held-out test set:")
            print(f"   python evaluate_regressor.py --holdout --model-path {fold_models[0]['model_path']}")
            print(f"   python evaluate_regressor.py --cv-ensemble --checkpoints-dir {args.checkpoints_dir}")
        
    elif (checkpoints_dir / "ensemble_summary.json").exists():
        print("Found ensemble training results")
        with open(checkpoints_dir / "ensemble_summary.json") as f:
            ensemble_summary = json.load(f)
        
        # use the best model from ensemble
        best_model_idx = np.argmin([m['best_val_mse'] for m in ensemble_summary['models']])
        best_model_path = ensemble_summary['models'][best_model_idx]['model_path']
        print(f"Using best ensemble model: {best_model_path}")
        
        evaluator = CLAMRegressorEvaluator(config_path, best_model_path)
        
        save_dir = "evaluation_results_regressor_ensemble"
        all_results = evaluator.evaluate_all_splits(save_dir)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {save_dir}")
        
        save_path = Path(save_dir)
        for split, results in all_results.items():
            results['predictions'].to_csv(save_path / f'{split}_detailed_predictions.csv', index=False)
            
    elif (checkpoints_dir / "best_model.pth").exists():
        best_model_path = checkpoints_dir / "best_model.pth"
        print(f"Using single best model: {best_model_path}")
        
        evaluator = CLAMRegressorEvaluator(config_path, best_model_path)
        
        save_dir = "evaluation_results_regressor"
        all_results = evaluator.evaluate_all_splits(save_dir)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {save_dir}")
        
        save_path = Path(save_dir)
        for split, results in all_results.items():
            results['predictions'].to_csv(save_path / f'{split}_detailed_predictions.csv', index=False)
    
    else:
        print(f"No trained models found in {checkpoints_dir}")
        print("Please train models first using train_regressor.py")


if __name__ == "__main__":
    main() 