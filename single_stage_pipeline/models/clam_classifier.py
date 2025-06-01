import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .clam_base import CLAM


class OrdinalLoss(nn.Module):
    # ordinal loss for tubulitis classification that respects the ordering T0 < T1 < T2 < T3
    
    def __init__(self, num_classes: int = 4):
        super(OrdinalLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        
        # create ordinal targets: [1,1,1] for class 3, [1,1,0] for class 2, [1,0,0] for class 1, [0,0,0] for class 0
        ordinal_targets = torch.zeros_like(logits)
        for i in range(batch_size):
            target_class = int(targets[i])
            if target_class > 0:
                ordinal_targets[i, :target_class] = 1.0
                
        return F.binary_cross_entropy_with_logits(logits, ordinal_targets)


class CLAMClassifier(nn.Module):
    
    def __init__(self, config: Dict[str, Any]):
        super(CLAMClassifier, self).__init__()
        
        model_config = config['model']
        
        self.clam = CLAM(
            input_dim=model_config['feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=model_config['num_classes'],  # will be 3 for ordinal (num_classes - 1)
            dropout=model_config['clam']['dropout'],
            k_sample=model_config['clam']['k_sample'],
            instance_loss_fn=model_config['clam']['instance_loss_fn'],
            gate=model_config['clam']['gate'],
            size_arg=model_config['clam']['size_arg']
        )
        
        self.bag_loss_weight = config['training']['bag_loss_weight']
        self.instance_loss_weight = config['training']['instance_loss_weight']
        self.ordinal_loss = OrdinalLoss(num_classes=4)
        
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        results = self.clam(features)
        
        if labels is not None:
            # ordinal loss (main bag-level loss)
            bag_loss = self.ordinal_loss(results['logits'], labels)
            
            # instance-level loss (for regularization)
            instance_loss = self.clam.calculate_instance_loss(results['instance_logits'], labels)
            
            # combined loss
            total_loss = (self.bag_loss_weight * bag_loss + 
                         self.instance_loss_weight * instance_loss)
            
            results.update({
                'loss': total_loss,
                'bag_loss': bag_loss,
                'instance_loss': instance_loss
            })
        
        return results
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            results = self.clam(features)
            cumulative_logits = results['logits']
            
            cumulative_probs = torch.sigmoid(cumulative_logits)
            
            class_probs = torch.zeros(cumulative_probs.size(0), 4, device=cumulative_probs.device)
            
            class_probs[:, 0] = 1 - cumulative_probs[:, 0]
            
            for k in range(1, 3):
                class_probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
            
            class_probs[:, 3] = cumulative_probs[:, 2]
            
            predictions = torch.argmax(class_probs, dim=1)
            
            assert predictions.shape[0] == cumulative_logits.shape[0], f"Prediction shape mismatch: {predictions.shape} vs {cumulative_logits.shape}"
            
        return predictions
    
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            results = self.clam(features)
            cumulative_logits = results['logits']
            cumulative_probs = torch.sigmoid(cumulative_logits)
            
            class_probs = torch.zeros(cumulative_probs.size(0), 4, device=cumulative_probs.device)
            class_probs[:, 0] = 1 - cumulative_probs[:, 0]
            
            for k in range(1, 3):
                class_probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
            
            class_probs[:, 3] = cumulative_probs[:, 2]
            
        return class_probs


def create_clam_classifier(config: Dict[str, Any]) -> CLAMClassifier:
    return CLAMClassifier(config)


def create_clam_model(config: Dict[str, Any]) -> CLAMClassifier:
    return CLAMClassifier(config) 