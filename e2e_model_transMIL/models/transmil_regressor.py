import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from e2e_model_transMIL.models.attention_modules import TransMILAggregator


class TransMILRegressor(nn.Module):
    # main regressor for tubulitis scoring using transformer-based multiple instance learning
    
    def __init__(self, config: Dict[str, Any]):
        super(TransMILRegressor, self).__init__()
        
        model_config = config['model']
        
        # core transformer aggregator
        self.aggregator = TransMILAggregator(
            input_dim=model_config['feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout']
        )
        
        # bag-level regressor
        self.bag_regressor = nn.Sequential(
            nn.Linear(model_config['hidden_dim'], model_config['hidden_dim'] // 2),
            nn.GELU(),
            nn.Dropout(model_config['dropout']),
            nn.Linear(model_config['hidden_dim'] // 2, 1)
        )
        
        # loss weights
        self.bag_loss_weight = config['training']['bag_loss_weight']
        self.instance_loss_weight = config['training']['instance_loss_weight']
        self.attention_reg_weight = config['training'].get('attention_reg_weight', 0.01)
        
        # tubulitis-specific parameters
        self.score_range = (0.0, 3.0)
        self.high_score_threshold = 2.0  # for instance pseudo-labeling
        
    def forward(self, features: torch.Tensor, coordinates: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        
        # transformer aggregation with attention
        aggregator_output = self.aggregator(features, coordinates)
        
        # bag-level prediction
        logits = self.bag_regressor(aggregator_output['bag_features'])
        predictions = torch.clamp(logits.squeeze(-1), *self.score_range)
        
        results = {
            'predictions': predictions,
            'attention_weights': aggregator_output['attention_weights'],
            'instance_logits': aggregator_output['instance_logits'],
            'patch_features': aggregator_output['patch_features'],
            'attention_maps': aggregator_output['attention_maps']
        }
        
        if labels is not None:
            # bag-level regression loss (huber loss for robustness)
            bag_loss = F.huber_loss(predictions, labels.float(), delta=1.0)
            
            # instance-level pseudo-labeling loss
            instance_loss = self._compute_instance_loss(
                aggregator_output['instance_logits'], 
                labels, 
                aggregator_output['attention_weights']
            )
            
            # attention regularization (encourage sparsity)
            attention_reg = self._compute_attention_regularization(
                aggregator_output['attention_weights']
            )
            
            # combined loss
            total_loss = (self.bag_loss_weight * bag_loss + 
                         self.instance_loss_weight * instance_loss +
                         self.attention_reg_weight * attention_reg)
            
            results.update({
                'loss': total_loss,
                'bag_loss': bag_loss,
                'instance_loss': instance_loss,
                'attention_reg': attention_reg
            })
        
        return results
    
    def _compute_instance_loss(self, instance_logits: torch.Tensor, 
                              bag_labels: torch.Tensor, 
                              attention_weights: torch.Tensor) -> torch.Tensor:
        # instance pseudo-labeling based on bag labels and attention weights
        batch_size, num_patches, num_classes = instance_logits.shape
        
        # create pseudo-labels: high tubulitis scores indicate relevant patches
        bag_labels_binary = (bag_labels >= self.high_score_threshold).float()
        
        # use attention weights to identify most relevant patches
        # top-k patches get positive labels for high-score bags
        k = min(max(1, num_patches // 10), 32)  # adaptive k based on patch count
        
        instance_labels = torch.zeros(batch_size, num_patches, device=instance_logits.device)
        
        for i, (bag_label, attention) in enumerate(zip(bag_labels_binary, attention_weights)):
            if bag_label > 0:  # positive bag
                # top-k attended patches are positive
                _, top_indices = torch.topk(attention, k)
                instance_labels[i, top_indices] = 1.0
            # negative bags remain all zeros
        
        # weighted cross-entropy (handle class imbalance)
        pos_weight = torch.tensor([1.0, 3.0], device=instance_logits.device)  # weight positive class more
        
        instance_labels_long = instance_labels.long()
        instance_loss = F.cross_entropy(
            instance_logits.view(-1, num_classes),
            instance_labels_long.view(-1),
            weight=pos_weight
        )
        
        return instance_loss
    
    def _compute_attention_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        # encourage sparse, focused attention on relevant tubule regions
        
        # entropy regularization (encourage sharpness)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        entropy_loss = entropy.mean()
        
        # l1 regularization (encourage sparsity)
        l1_loss = attention_weights.abs().mean()
        
        return 0.5 * entropy_loss + 0.5 * l1_loss
    
    def predict(self, features: torch.Tensor, coordinates: torch.Tensor) -> Dict[str, Any]:
        # inference mode with attention visualization
        self.eval()
        with torch.no_grad():
            results = self.forward(features, coordinates)
            
            # additional processing for visualization
            attention_weights = results['attention_weights']
            predictions = results['predictions']
            
            # identify top attended patches for visualization
            top_k = min(10, attention_weights.shape[-1])
            top_attention_values, top_attention_indices = torch.topk(
                attention_weights, top_k, dim=-1
            )
            
            results.update({
                'top_attention_indices': top_attention_indices,
                'top_attention_values': top_attention_values,
                'predicted_scores': predictions.cpu().numpy()
            })
            
        return results


def create_transmil_regressor(config: Dict[str, Any]) -> TransMILRegressor:
    return TransMILRegressor(config) 