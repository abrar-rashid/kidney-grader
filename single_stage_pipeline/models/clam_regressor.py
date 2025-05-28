import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .clam_base import CLAM


class CLAMRegressor(nn.Module): # regression model for the regression problem formulation as mentioned in paper
    # gives continuous tubulitis score predictions, which are clamped to the range [0, 3]
    
    def __init__(self, config: Dict[str, Any]):
        super(CLAMRegressor, self).__init__()
        
        model_config = config['model']
        
        self.clam = CLAM(
            input_dim=model_config['feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=model_config['num_classes'],
            dropout=model_config['clam']['dropout'],
            k_sample=model_config['clam']['k_sample'],
            instance_loss_fn=model_config['clam']['instance_loss_fn'],
            gate=model_config['clam']['gate'],
            size_arg=model_config['clam']['size_arg']
        )
        
        self.bag_loss_weight = config['training']['bag_loss_weight']
        self.instance_loss_weight = config['training']['instance_loss_weight']
        
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        results = self.clam(features)
        
        if labels is not None:
            # bag-level loss (MSE for regression)
            bag_loss = F.mse_loss(results['logits'].squeeze(), labels.float())
            
            # instance-level loss
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
            predictions = results['logits'].squeeze()
            
            # clamp predictions to valid range [0, 3]
            predictions = torch.clamp(predictions, 0.0, 3.0)
            
        return predictions


def create_clam_regressor(config: Dict[str, Any]) -> CLAMRegressor:
    return CLAMRegressor(config)


def create_clam_model(config: Dict[str, Any]) -> CLAMRegressor:
    return CLAMRegressor(config) 