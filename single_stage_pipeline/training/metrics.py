import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    cohen_kappa_score, mean_absolute_error, classification_report
)
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings("ignore")


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    try:
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    except:
        return 0.0


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.abs(y_true - y_pred)
    return np.mean(diff <= 1).item()


def kendall_tau_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        tau, _ = kendalltau(y_true, y_pred)
        return tau if not np.isnan(tau) else 0.0
    except:
        return 0.0


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.5) -> Dict[str, float]:
    # evaluate binary classification: low-grade (T0,T1) vs high-grade (T2,T3)
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary', zero_division=0
    )
    
    return {
        'binary_accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'binary_precision': precision,
        'binary_recall': recall,
        'binary_f1': f1,
        'high_grade_sensitivity': recall,
        'low_grade_specificity': precision 
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> Dict[str, Any]:
    if class_names is None:
        class_names = [f"T{i}" for i in range(4)]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(4)
    )
    
    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[f'{class_name}_precision'] = precision[i] if i < len(precision) else 0.0
        per_class[f'{class_name}_recall'] = recall[i] if i < len(recall) else 0.0
        per_class[f'{class_name}_f1'] = f1[i] if i < len(f1) else 0.0
        per_class[f'{class_name}_support'] = int(support[i]) if i < len(support) else 0
    
    return per_class


def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=range(4))
    
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    return {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'main_diagonal_accuracy': np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    }


def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: List[str] = None,
    binary_threshold: float = 1.5
) -> Dict[str, Any]:
    
    if class_names is None:
        class_names = ["T0", "T1", "T2", "T3"]
    
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(4)
    )
    
    metrics['macro_f1'] = np.mean(f1)
    metrics['weighted_f1'] = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[2]
    
    # ordinal-specific
    metrics['quadratic_kappa'] = quadratic_weighted_kappa(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['adjacent_accuracy'] = adjacent_accuracy(y_true, y_pred)
    metrics['kendall_tau'] = kendall_tau_correlation(y_true, y_pred)
    
    metrics.update(per_class_metrics(y_true, y_pred, class_names))
    
    metrics.update(binary_classification_metrics(y_true, y_pred, binary_threshold))
    
    metrics.update(confusion_matrix_metrics(y_true, y_pred))
    
    return metrics


def get_metric_for_monitoring(metrics: Dict[str, Any], monitor_metric: str) -> float:
    return metrics.get(monitor_metric, 0.0)


def format_metrics_for_logging(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    scalar_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            scalar_metrics[f"{prefix}{key}"] = float(value)
    
    return scalar_metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None):
    if class_names is None:
        class_names = ["T0", "T1", "T2", "T3"]
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    metrics = calculate_classification_metrics(y_true, y_pred, class_names)
    
    print(f"\nAdditional Metrics:")
    print(f"Quadratic Weighted Kappa: {metrics['quadratic_kappa']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"Adjacent Accuracy (±1): {metrics['adjacent_accuracy']:.4f}")
    print(f"Kendall's Tau: {metrics['kendall_tau']:.4f}")
    
    print(f"\nBinary Classification (T0,T1 vs T2,T3):")
    print(f"Binary Accuracy: {metrics['binary_accuracy']:.4f}")
    print(f"High-grade Sensitivity: {metrics['high_grade_sensitivity']:.4f}")
    print(f"Low-grade Specificity: {metrics['low_grade_specificity']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print("     ", "  ".join(class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:4s}", "  ".join([f"{x:3d}" for x in row]))
    
    print("="*50)


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   class_names: List[str] = None, binary_threshold: float = 1.5) -> Dict[str, Any]:
    return calculate_classification_metrics(
        targets.cpu().numpy(), 
        predictions.cpu().numpy(), 
        class_names, 
        binary_threshold
    ) 