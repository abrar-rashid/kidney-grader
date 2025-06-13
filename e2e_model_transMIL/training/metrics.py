import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple


class TubulitisMetrics:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.slide_names = []
        
    def update(self, predictions: np.ndarray, targets: np.ndarray, slide_names: List[str] = None):
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        predictions = np.clip(predictions, 0.0, 3.0)
        
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        
        if slide_names:
            self.slide_names.extend(slide_names)
            
    def compute(self) -> Dict[str, float]:
        
        if len(self.predictions) == 0:
            return {}
            
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        
        metrics['r2'] = r2_score(targets, predictions)
        
        try:
            pearson_corr, pearson_p = pearsonr(predictions, targets)
            metrics['pearson_correlation'] = pearson_corr
            metrics['pearson_p_value'] = pearson_p
        except:
            metrics['pearson_correlation'] = 0.0
            metrics['pearson_p_value'] = 1.0
        
        try:
            spearman_corr, spearman_p = spearmanr(predictions, targets)
            metrics['spearman_correlation'] = spearman_corr
            metrics['spearman_p_value'] = spearman_p
        except:
            metrics['spearman_correlation'] = 0.0
            metrics['spearman_p_value'] = 1.0
        
        metrics.update(self._compute_tubulitis_metrics(predictions, targets))
        
        return metrics
    
    def _compute_tubulitis_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        
        metrics = {}
        
        rounded_predictions = np.round(predictions)
        rounded_targets = np.round(targets)
        
        metrics['discrete_accuracy'] = np.mean(rounded_predictions == rounded_targets)
        
        metrics['within_1_accuracy'] = np.mean(np.abs(predictions - targets) <= 1.0)
        
        for score in [0, 1, 2, 3]:
            mask = rounded_targets == score
            if mask.sum() > 0:
                class_mae = np.mean(np.abs(predictions[mask] - targets[mask]))
                metrics[f'mae_score_{int(score)}'] = class_mae
                
                class_acc = np.mean(rounded_predictions[mask] == rounded_targets[mask])
                metrics[f'accuracy_score_{int(score)}'] = class_acc
        
        binary_targets = (targets >= 2.0).astype(int)
        binary_predictions = (predictions >= 2.0).astype(int)
        
        metrics['binary_accuracy'] = np.mean(binary_targets == binary_predictions)
        
        if binary_targets.sum() > 0:
            true_positives = np.sum((binary_targets == 1) & (binary_predictions == 1))
            false_positives = np.sum((binary_targets == 0) & (binary_predictions == 1))
            false_negatives = np.sum((binary_targets == 1) & (binary_predictions == 0))
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics['binary_precision'] = precision
            metrics['binary_recall'] = recall
            metrics['binary_f1'] = f1
        
        errors = predictions - targets
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_absolute_error'] = np.max(np.abs(errors))
        
        for tolerance in [0.5, 1.0, 1.5]:
            within_tolerance = np.mean(np.abs(errors) <= tolerance)
            metrics[f'within_{tolerance}_tolerance'] = within_tolerance
            
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        
        if len(self.predictions) == 0:
            return np.zeros((4, 4))
            
        predictions = np.round(np.array(self.predictions)).astype(int)
        targets = np.round(np.array(self.targets)).astype(int)
        
        predictions = np.clip(predictions, 0, 3)
        targets = np.clip(targets, 0, 3)
        
        confusion = np.zeros((4, 4), dtype=int)
        for t, p in zip(targets, predictions):
            confusion[t, p] += 1
            
        return confusion
    
    def get_per_slide_results(self) -> List[Dict]:
        
        if not self.slide_names or len(self.slide_names) != len(self.predictions):
            return []
            
        results = []
        for slide, pred, target in zip(self.slide_names, self.predictions, self.targets):
            results.append({
                'slide_name': slide,
                'prediction': pred,
                'target': target,
                'error': pred - target,
                'absolute_error': abs(pred - target),
                'discrete_prediction': round(pred),
                'discrete_target': round(target),
                'discrete_correct': round(pred) == round(target)
            })
            
        return results


def compute_fold_metrics(all_predictions: List[np.ndarray], 
                        all_targets: List[np.ndarray],
                        all_slide_names: List[List[str]] = None) -> Dict[str, Dict]:
    
    fold_metrics = {}
    
    for fold_idx, (preds, targets) in enumerate(zip(all_predictions, all_targets)):
        metrics_calculator = TubulitisMetrics()
        slide_names = all_slide_names[fold_idx] if all_slide_names else None
        metrics_calculator.update(preds, targets, slide_names)
        
        fold_metrics[f'fold_{fold_idx}'] = metrics_calculator.compute()
    
    overall_metrics = {}
    metric_names = list(fold_metrics['fold_0'].keys())
    
    for metric_name in metric_names:
        values = [fold_metrics[f'fold_{i}'][metric_name] for i in range(len(all_predictions))]
        overall_metrics[f'{metric_name}_mean'] = np.mean(values)
        overall_metrics[f'{metric_name}_std'] = np.std(values)
        overall_metrics[f'{metric_name}_min'] = np.min(values)
        overall_metrics[f'{metric_name}_max'] = np.max(values)
    
    return {
        'fold_metrics': fold_metrics,
        'overall_metrics': overall_metrics
    } 