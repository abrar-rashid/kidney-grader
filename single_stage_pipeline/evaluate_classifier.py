import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from training.dataset import create_data_loaders, WSIFeaturesDataset, collate_fn
from training.metrics import (
    compute_metrics, print_classification_report, 
    format_metrics_for_logging, confusion_matrix_metrics
)

warnings.filterwarnings("ignore")


class CLAMClassifierEvaluator:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        self.class_names = self.config.get('classification', {}).get('class_names', ["T0", "T1", "T2", "T3"])
        self.binary_threshold = self.config.get('classification', {}).get('binary_threshold', 1.5)
        
    def load_model(self, model_path: str):
        from models.clam_classifier import create_clam_classifier
        
        model = create_clam_classifier(self.config)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_single_model(self, model_path: str, data_loader, 
                            save_predictions: bool = True, 
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        
        model = self.load_model(model_path)
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_slide_names = []
        
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                slide_names = batch['slide_names']
                
                predictions = model.predict(features)
                probabilities = model.predict_proba(features)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_slide_names.extend(slide_names)
        
        metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets),
            self.class_names,
            self.binary_threshold
        )
        
        if save_predictions and output_dir:
            self._save_predictions(
                all_slide_names, all_targets, all_predictions, 
                all_probabilities, output_dir
            )
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'slide_names': all_slide_names
        }
    
    def evaluate_ensemble(self, model_paths: List[str], data_loader,
                         save_predictions: bool = True,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        
        all_model_predictions = []
        all_model_probabilities = []
        all_targets = None
        all_slide_names = None
        
        for i, model_path in enumerate(model_paths):
            print(f"Evaluating model {i+1}/{len(model_paths)}: {model_path}")
            
            result = self.evaluate_single_model(
                model_path, data_loader, save_predictions=False
            )
            
            all_model_predictions.append(result['predictions'])
            all_model_probabilities.append(result['probabilities'])
            
            if all_targets is None:
                all_targets = result['targets']
                all_slide_names = result['slide_names']
        
        ensemble_predictions = []
        ensemble_probabilities = []
        
        for i in range(len(all_targets)):
            # get predictions from all models for this sample
            sample_preds = [preds[i] for preds in all_model_predictions]
            sample_probs = [probs[i] for probs in all_model_probabilities]
            
            # majority vote for final prediction
            ensemble_pred = max(set(sample_preds), key=sample_preds.count)
            ensemble_predictions.append(ensemble_pred)
            
            # average probabilities
            ensemble_prob = np.mean(sample_probs, axis=0)
            ensemble_probabilities.append(ensemble_prob)
        
        metrics = compute_metrics(
            torch.tensor(ensemble_predictions),
            torch.tensor(all_targets),
            self.class_names,
            self.binary_threshold
        )
        
        if save_predictions and output_dir:
            self._save_predictions(
                all_slide_names, all_targets, ensemble_predictions,
                ensemble_probabilities, output_dir, prefix="ensemble_"
            )
        
        return {
            'metrics': metrics,
            'predictions': ensemble_predictions,
            'probabilities': ensemble_probabilities,
            'targets': all_targets,
            'slide_names': all_slide_names,
            'individual_predictions': all_model_predictions,
            'individual_probabilities': all_model_probabilities
        }
    
    def evaluate_cross_validation(self, cv_dir: str, n_folds: int = 5) -> Dict[str, Any]:
        cv_results = []
        all_predictions = []
        all_targets = []
        
        for fold in range(n_folds):
            fold_dir = Path(cv_dir) / f"cv_fold_{fold}"
            model_path = fold_dir / "best_model.pth"
            
            if not model_path.exists():
                print(f"Warning: Model not found for fold {fold}: {model_path}")
                continue
            
            features_dir = self.config['data']['features_dir']
            labels_file = self.config['data']['labels_file']
            split_file = Path(features_dir) / 'splits' / f'cv_fold_{fold}_splits.json'
            
            if not split_file.exists():
                print(f"Warning: Split file not found for fold {fold}: {split_file}")
                continue
            
            _, val_loader, _ = create_data_loaders(
                features_dir, labels_file, str(split_file), self.config
            )
            
            result = self.evaluate_single_model(
                str(model_path), val_loader, save_predictions=False
            )
            
            cv_results.append({
                'fold': fold,
                'metrics': format_metrics_for_logging(result['metrics'])
            })
            
            all_predictions.extend(result['predictions'])
            all_targets.extend(result['targets'])
        
        overall_metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets),
            self.class_names,
            self.binary_threshold
        )
        
        return {
            'fold_results': cv_results,
            'overall_metrics': overall_metrics,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def _save_predictions(self, slide_names: List[str], targets: List[int], 
                         predictions: List[int], probabilities: List[np.ndarray],
                         output_dir: str, prefix: str = ""):
        
        os.makedirs(output_dir, exist_ok=True)
        
        results_data = {
            'slide_name': slide_names,
            'true_label': targets,
            'predicted_label': predictions,
            'true_class': [self.class_names[int(t)] for t in targets],
            'predicted_class': [self.class_names[int(p)] for p in predictions]
        }
        
        for i, class_name in enumerate(self.class_names):
            results_data[f'prob_{class_name}'] = [prob[i] for prob in probabilities]
        
        results_df = pd.DataFrame(results_data)
        
        csv_path = Path(output_dir) / f"{prefix}predictions.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
        
        return results_df
    
    def plot_confusion_matrix(self, targets: List[int], predictions: List[int],
                            output_dir: str, prefix: str = ""):
        
        os.makedirs(output_dir, exist_ok=True)
        
        cm = confusion_matrix(targets, predictions, labels=range(4))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        plot_path = Path(output_dir) / f"{prefix}confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {plot_path}")
    
    def plot_class_distribution(self, targets: List[int], predictions: List[int],
                              output_dir: str, prefix: str = ""):
        
        os.makedirs(output_dir, exist_ok=True)
        
        true_counts = [targets.count(i) for i in range(4)]
        pred_counts = [predictions.count(i) for i in range(4)]
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        
        ax.set_xlabel('Tubulitis Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution: True vs Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        
        plt.tight_layout()
        
        plot_path = Path(output_dir) / f"{prefix}class_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution plot saved to: {plot_path}")


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


def evaluate_on_holdout_test(model_path: str, config_path: str, output_dir: str = None) -> Dict[str, Any]:
    
    print("="*60)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    features_dir = config['data']['features_dir']
    holdout_file = Path(features_dir) / 'splits' / 'holdout_split.json'
    
    if not holdout_file.exists():
        raise FileNotFoundError(f"Holdout split file not found: {holdout_file}")
    
    with open(holdout_file, 'r') as f:
        holdout_split = json.load(f)
    
    test_slides = holdout_split['test']
    print(f"Evaluating on {len(test_slides)} held-out test slides")
    
    test_split = {
        'train': [],
        'val': [],
        'test': test_slides
    }
    
    temp_split_file = Path(features_dir) / 'splits' / 'temp_holdout_test.json'
    with open(temp_split_file, 'w') as f:
        json.dump(test_split, f, indent=2)
    
    try:
        test_loader = create_test_only_loader(
            features_dir, config['data']['labels_file'], str(temp_split_file), config
        )
        
        from models.clam_classifier import create_clam_classifier
        device = torch.device(config['hardware']['device'])
        model = create_clam_classifier(config).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Model trained for {checkpoint['epoch']} epochs")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        slide_names = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                preds = model.predict(features)
                probs = model.predict_proba(features)
                
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                probs_np = probs.cpu().numpy()
                
                if len(preds_np.shape) == 0:
                    preds_np = [preds_np.item()]
                    labels_np = [labels_np.item()]
                    probs_np = [probs_np]
                elif len(preds_np.shape) == 1:
                    preds_np = preds_np.tolist()
                    labels_np = labels_np.tolist()
                    probs_np = probs_np.tolist()
                
                all_predictions.extend(preds_np)
                all_targets.extend(labels_np)
                all_probabilities.extend(probs_np)
                slide_names.extend(batch['slide_names'])
        
        class_names = config.get('classification', {}).get('class_names', ["T0", "T1", "T2", "T3"])
        binary_threshold = config.get('classification', {}).get('binary_threshold', 1.5)
        
        metrics = compute_metrics(
            torch.tensor(all_predictions), 
            torch.tensor(all_targets),
            class_names,
            binary_threshold
        )
        
        results_df = pd.DataFrame({
            'slide_name': slide_names,
            'prediction': all_predictions,
            'target': all_targets,
            'T0_prob': [p[0] for p in all_probabilities],
            'T1_prob': [p[1] for p in all_probabilities],
            'T2_prob': [p[2] for p in all_probabilities],
            'T3_prob': [p[3] for p in all_probabilities]
        })
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_df.to_csv(output_path / 'holdout_test_results.csv', index=False)
            
            with open(output_path / 'holdout_test_metrics.json', 'w') as f:
                json.dump(format_metrics_for_logging(metrics), f, indent=2)
            
            print(f"Results saved to: {output_path}")
        
        print(f"\n{'='*50}")
        print("HELD-OUT TEST SET RESULTS")
        print(f"{'='*50}")
        
        formatted_metrics = format_metrics_for_logging(metrics)
        for metric, value in formatted_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print_classification_report(
            np.array(all_targets), 
            np.array(all_predictions),
            class_names
        )
        
        return {
            'metrics': formatted_metrics,
            'predictions': results_df,
            'num_samples': len(all_predictions)
        }
        
    finally:
        if temp_split_file.exists():
            temp_split_file.unlink()


def evaluate_cv_ensemble_on_holdout(cv_dir: str, config_path: str, output_dir: str = None) -> Dict[str, Any]:
    
    print("="*80)
    print("CLAM Tubulitis Classification - Held-Out Test Set Evaluation")
    print("="*80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cv_path = Path(cv_dir)
    fold_models = []
    
    for fold_dir in sorted(cv_path.glob("cv_fold_*")):
        best_model = fold_dir / "best_model.pth"
        if best_model.exists():
            fold_models.append(str(best_model))
            print(f"Found model: {best_model}")
        else:
            print(f"Warning: No model found in {fold_dir}")
    
    if not fold_models:
        raise FileNotFoundError(f"No CV fold models found in {cv_dir}")
    
    print(f"\nLoaded {len(fold_models)} models for ensemble evaluation")
    
    features_dir = config['data']['features_dir']
    holdout_file = Path(features_dir) / 'splits' / 'holdout_split.json'
    
    with open(holdout_file, 'r') as f:
        holdout_split = json.load(f)
    
    test_slides = holdout_split['test']
    print(f"Test slides: {len(test_slides)}")
    
    test_split = {
        'train': [],
        'val': [],
        'test': test_slides
    }
    
    temp_split_file = Path(features_dir) / 'splits' / 'temp_holdout_test.json'
    with open(temp_split_file, 'w') as f:
        json.dump(test_split, f, indent=2)
    
    try:
        test_loader = create_test_only_loader(
            features_dir, config['data']['labels_file'], str(temp_split_file), config
        )
        
        from models.clam_classifier import create_clam_classifier
        device = torch.device(config['hardware']['device'])
        models = []
        
        for model_path in fold_models:
            model = create_clam_classifier(config).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        
        print(f"Loaded {len(models)} models for ensemble")
        
        all_fold_predictions = []
        all_targets = []
        slide_names = []
        
        print("\nEvaluating individual folds...")
        for fold_idx, (model, model_path) in enumerate(zip(models, fold_models)):
            fold_predictions = []
            fold_targets = []
            fold_slide_names = []
            
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(device)
                    labels = batch['labels'].to(device)
                    
                    probs = model.predict_proba(features)
                    preds = torch.argmax(probs, dim=-1).cpu().numpy()
                    
                    if len(preds.shape) == 0:
                        preds = [preds.item()]
                    
                    fold_predictions.extend(preds)
                    
                    if fold_idx == 0: 
                        labels_np = labels.cpu().numpy()
                        if len(labels_np.shape) == 0:
                            labels_np = [labels_np.item()]
                        fold_targets.extend(labels_np)
                        fold_slide_names.extend(batch['slide_names'])
            
            all_fold_predictions.append(np.array(fold_predictions))
            if fold_idx == 0:
                all_targets = np.array(fold_targets)
                slide_names = fold_slide_names
        
        all_fold_predictions = np.array(all_fold_predictions) 
        ensemble_predictions = []
        
        for i in range(len(all_targets)):
            sample_preds = all_fold_predictions[:, i]
            unique, counts = np.unique(sample_preds, return_counts=True)
            ensemble_pred = unique[np.argmax(counts)]
            ensemble_predictions.append(ensemble_pred)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        class_names = config.get('classification', {}).get('class_names', ["T0", "T1", "T2", "T3"])
        binary_threshold = config.get('classification', {}).get('binary_threshold', 1.5)
        
        ensemble_metrics = compute_metrics(
            torch.tensor(ensemble_predictions), 
            torch.tensor(all_targets),
            class_names,
            binary_threshold
        )
        
        per_fold_metrics = []
        for fold_idx, fold_preds in enumerate(all_fold_predictions):
            fold_metrics = compute_metrics(
                torch.tensor(fold_preds), 
                torch.tensor(all_targets),
                class_names,
                binary_threshold
            )
            per_fold_metrics.append({
                'fold': fold_idx,
                'accuracy': fold_metrics['accuracy'],
                'macro_f1': fold_metrics['macro_f1'],
                'quadratic_kappa': fold_metrics['quadratic_kappa'],
                'mae': fold_metrics['mae'],
                'binary_accuracy': fold_metrics['binary_accuracy']
            })
        
        ensemble_results_df = pd.DataFrame({
            'slide_name': slide_names,
            'ensemble_prediction': ensemble_predictions,
            'target': all_targets,
            'correct': ensemble_predictions == all_targets,
            'abs_error': np.abs(ensemble_predictions - all_targets)
        })
        
        cm = confusion_matrix(all_targets, ensemble_predictions, labels=[0, 1, 2, 3])
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            ensemble_results_df.to_csv(output_path / 'holdout_ensemble_results_classifier.csv', index=False)
            
            with open(output_path / 'holdout_ensemble_metrics_classifier.json', 'w') as f:
                json.dump(format_metrics_for_logging(ensemble_metrics), f, indent=2)
            
            np.savetxt(output_path / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
        
        from datetime import datetime
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("CLAM Tubulitis Classification - Held-Out Test Set Evaluation")
        report_lines.append("="*80)
        report_lines.append("")
        
        report_lines.append(f"Configuration:")
        report_lines.append(f"  Test slides: {len(test_slides)}")
        report_lines.append(f"  Successfully evaluated: {len(slide_names)}")
        report_lines.append(f"  Model folds: {len(models)}")
        report_lines.append(f"  Target column: T")
        report_lines.append("")
        
        report_lines.append(f"ENSEMBLE PERFORMANCE:")
        report_lines.append(f"{'='*40}")
        formatted_metrics = format_metrics_for_logging(ensemble_metrics)
        key_metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'quadratic_kappa', 'mae', 
                      'adjacent_accuracy', 'kendall_tau', 'binary_accuracy', 
                      'high_grade_sensitivity', 'low_grade_specificity']
        
        for metric in key_metrics:
            if metric in formatted_metrics:
                report_lines.append(f"  {metric.replace('_', ' ').title()}: {formatted_metrics[metric]:.4f}")
        report_lines.append("")
        
        report_lines.append(f"PER-FOLD PERFORMANCE:")
        report_lines.append(f"{'='*40}")
        for fold_metrics in per_fold_metrics:
            report_lines.append(f"Fold {fold_metrics['fold']} ({len(slide_names)} slides):")
            report_lines.append(f"  Accuracy: {fold_metrics['accuracy']:.4f}, Macro F1: {fold_metrics['macro_f1']:.4f}")
            report_lines.append(f"  Quadratic Kappa: {fold_metrics['quadratic_kappa']:.4f}, MAE: {fold_metrics['mae']:.4f}")
            report_lines.append(f"  Binary Accuracy: {fold_metrics['binary_accuracy']:.4f}")
            report_lines.append("")
        
        report_lines.append(f"PER-SLIDE RESULTS (ENSEMBLE):")
        report_lines.append(f"{'='*70}")
        report_lines.append(f"{'Slide':<40} {'True':>6} {'Pred':>6} {'Correct':>8} {'|Error|':>8}")
        report_lines.append(f"{'='*70}")
        for _, row in ensemble_results_df.iterrows():
            slide_short = row['slide_name'][-36:] if len(row['slide_name']) > 36 else row['slide_name']
            correct_str = "✓" if row['correct'] else "✗"
            report_lines.append(f"{slide_short:<40} T{int(row['target'])}     T{int(row['ensemble_prediction'])}     {correct_str:>6} {row['abs_error']:>8.0f}")
        
        report_lines.append("")
        report_lines.append(f"Confusion Matrix (T Scores):")
        report_lines.append(f"{'='*40}")
        report_lines.append("     T0  T1  T2  T3")
        for i, row in enumerate(cm):
            report_lines.append(f"T{i}  {row[0]:3d} {row[1]:3d} {row[2]:3d} {row[3]:3d}")
        
        report_lines.append("")
        report_lines.append("DETAILED CLASSIFICATION REPORT:")
        report_lines.append("="*50)
        
        from sklearn.metrics import classification_report
        class_report = classification_report(all_targets, ensemble_predictions, 
                                           target_names=class_names, zero_division=0)
        report_lines.extend(class_report.split('\n'))
        
        report_lines.append("")
        report_lines.append(f"Additional Metrics:")
        report_lines.append(f"  Quadratic Weighted Kappa: {ensemble_metrics['quadratic_kappa']:.4f}")
        report_lines.append(f"  Mean Absolute Error: {ensemble_metrics['mae']:.4f}")
        report_lines.append(f"  Adjacent Accuracy (±1): {ensemble_metrics['adjacent_accuracy']:.4f}")
        report_lines.append(f"  Kendall's Tau: {ensemble_metrics['kendall_tau']:.4f}")
        
        report_lines.append("")
        report_lines.append(f"Binary Classification (T0,T1 vs T2,T3):")
        report_lines.append(f"  Binary Accuracy: {ensemble_metrics['binary_accuracy']:.4f}")
        report_lines.append(f"  High-grade Sensitivity: {ensemble_metrics['high_grade_sensitivity']:.4f}")
        report_lines.append(f"  Low-grade Specificity: {ensemble_metrics['low_grade_specificity']:.4f}")
        
        report_lines.append("")
        report_lines.append(f"Report generated: {datetime.now()}")
        
        for line in report_lines:
            print(line)
        
        if output_dir:
            with open(output_path / 'evaluation_report.txt', 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"\nReport saved to: {output_path / 'evaluation_report.txt'}")
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['T0', 'T1', 'T2', 'T3'],
                       yticklabels=['T0', 'T1', 'T2', 'T3'],
                       cbar_kws={'label': 'Number of Samples'})
            plt.xlabel('Predicted Class', fontsize=14)
            plt.ylabel('True Class', fontsize=14)
            plt.title('Test - Confusion Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix heatmap saved to: {output_path / 'confusion_matrix_heatmap.png'}")
        
        return {
            'ensemble_metrics': format_metrics_for_logging(ensemble_metrics),
            'ensemble_predictions': ensemble_results_df,
            'per_fold_metrics': per_fold_metrics,
            'confusion_matrix': cm,
            'num_models': len(models),
            'num_samples': len(ensemble_predictions)
        }
        
    finally:
        if temp_split_file.exists():
            temp_split_file.unlink()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CLAM Classifier')
    parser.add_argument('--config', type=str, 
                        default='configs/clam_classifier_training.yaml',
                        help='Path to config file')
    parser.add_argument('--model-path', type=str,
                        help='Path to trained model checkpoint')
    parser.add_argument('--split-file', type=str,
                        help='Path to data splits JSON file')
    parser.add_argument('--ensemble-dir', type=str,
                        help='Directory containing ensemble models')
    parser.add_argument('--cv-dir', type=str,
                        help='Directory containing CV fold models')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save evaluation results')
    parser.add_argument('--holdout', action='store_true',
                        help='Evaluate on held-out test set')
    parser.add_argument('--cv-ensemble', action='store_true',
                        help='Evaluate CV ensemble on held-out test set')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    
    args = parser.parse_args()
    
    if args.holdout:
        if not args.model_path:
            raise ValueError("--model-path required for held-out evaluation")
        evaluate_on_holdout_test(args.model_path, args.config, args.output_dir)
    
    elif args.cv_ensemble:
        if not args.cv_dir:
            raise ValueError("--cv-dir required for CV ensemble evaluation")
        evaluate_cv_ensemble_on_holdout(args.cv_dir, args.config, args.output_dir)
    
    elif args.ensemble_dir:
        evaluate_ensemble(args.ensemble_dir, args.config, args.split_file, 
                         args.output_dir, args.visualize)
    
    elif args.model_path:
        evaluate_single_model(args.model_path, args.config, args.split_file, 
                             args.output_dir, args.visualize)
    
    else:
        print("Please specify --model-path, --ensemble-dir, --holdout, or --cv-ensemble")


if __name__ == "__main__":
    main() 