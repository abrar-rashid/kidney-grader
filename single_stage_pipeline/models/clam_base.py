import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional


class CLAMAttention(nn.Module): # stands for Clustering-constrained Attention Multiple Instance Learning
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.25, num_heads: int = 1):
        super(CLAMAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # features: [batch_size, num_instances, feature_dim]
        # returns (attention_weights: [batch_size, num_instances, num_heads], attended_features: [batch_size, num_heads, feature_dim])

        batch_size, num_instances, feature_dim = features.shape
        
        attention_weights = self.attention(features)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # apply attention (https://arxiv.org/pdf/1706.03762)
        attended_features = torch.bmm(
            attention_weights.transpose(1, 2),
            features
        )
        
        return attention_weights, attended_features


class CLAMGatedAttention(nn.Module):    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.25, num_heads: int = 1):
        super(CLAMGatedAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # same shapes as CLAMAttention forward params
        
        batch_size, num_instances, feature_dim = features.shape
        
        # gated attention mechanism
        V = self.attention_V(features)
        U = self.attention_U(features)
        
        # element-wise multiplication (gating)
        gated = V * U
        gated = self.dropout(gated)
        
        attention_weights = self.attention_weights(gated)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        attended_features = torch.bmm(
            attention_weights.transpose(1, 2),  # [batch_size, num_heads, num_instances]
            features  # [batch_size, num_instances, feature_dim]
        )  # [batch_size, num_heads, feature_dim]
        
        return attention_weights, attended_features


class CLAM(nn.Module):  # base implem. for CLAM
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.25,
        k_sample: int = 8,
        instance_loss_fn: str = "svm",
        gate: bool = True,
        size_arg: str = "small"
    ):
        super(CLAM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.gate = gate
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # attention mechanism, either gated or standard
        if gate:
            self.attention = CLAMGatedAttention(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout,
                num_heads=1
            )
        else:
            self.attention = CLAMAttention(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout,
                num_heads=1
            )
        
        # for bag-level prediction
        if size_arg == "small":
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        # instance-level classifier for regularization
        self.instance_classifier = nn.Linear(hidden_dim, 2)  # binary classification for instances
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, features: torch.Tensor, return_features: bool = False) -> Dict[str, Any]:
        ''' returns dictionary containing:
            logits: bag-level predictions [batch_size, num_classes]
            attention_weights: attention weights [batch_size, num_instances, 1]
            instance_logits: instance-level predictions [batch_size, k_sample, 2]
            selected_features: features of selected instances [batch_size, k_sample, hidden_dim] '''

        batch_size, num_instances, feature_dim = features.shape
        
        h = self.feature_extractor(features)
        
        # attention mechanism
        attention_weights, attended_features = self.attention(h)
        
        # bag-level prediction
        bag_features = attended_features.squeeze(1)
        logits = self.classifier(bag_features)
        
        # instance-level predictions for top-k instances
        top_k_indices = self._get_top_k_instances(attention_weights.squeeze(-1), self.k_sample)
        selected_features = self._select_features(h, top_k_indices) 
        instance_logits = self.instance_classifier(selected_features)
        
        results = {
            'logits': logits,
            'attention_weights': attention_weights,
            'instance_logits': instance_logits,
            'selected_features': selected_features
        }
        
        if return_features:
            results['transformed_features'] = h
            results['attended_features'] = attended_features
        
        return results
    
    def _get_top_k_instances(self, attention_weights: torch.Tensor, k: int) -> torch.Tensor:
        # get indiices of top-k instances based on attention weights
        batch_size, num_instances = attention_weights.shape
        k = min(k, num_instances)
        
        _, top_k_indices = torch.topk(attention_weights, k, dim=1)  # [batch_size, k]
        return top_k_indices
    
    def _select_features(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # select features based on indices
        batch_size, k = indices.shape
        _, num_instances, feature_dim = features.shape
        
        # expand indices for gathering
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        selected_features = torch.gather(features, 1, expanded_indices)
        
        return selected_features
    
    def calculate_instance_loss(self, instance_logits: torch.Tensor, bag_labels: torch.Tensor) -> torch.Tensor:
        # calculate instance-level loss for regularization
        batch_size, k_sample, num_instance_classes = instance_logits.shape
        
        '''Create binary pseudo labels for instances based on bag labels. For regression, 
            we can use a threshold to create binary pseudo labels. Assuming tubulitis 
            scores: 0-1 -> negative, 2-3 -> positive'''
        bag_labels_binary = (bag_labels >= 2.0).float()
        
        # expand bag labels to instance level
        instance_labels = bag_labels_binary.unsqueeze(1).expand(-1, k_sample).long()
        
        # cross-entropy loss
        instance_logits_flat = instance_logits.view(-1, num_instance_classes)  # [batch_size * k_sample, 2]
        instance_labels_flat = instance_labels.view(-1)  # [batch_size * k_sample]
        
        if self.instance_loss_fn == "ce":
            instance_loss = F.cross_entropy(instance_logits_flat, instance_labels_flat)
        elif self.instance_loss_fn == "svm":
            instance_loss = self._svm_loss(instance_logits_flat, instance_labels_flat)
        else:
            raise ValueError(f"Unknown instance loss function: {self.instance_loss_fn}")
        
        return instance_loss
    
    def _svm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # SVM-style hinge loss
        margin = 1.0
        scores = logits.gather(1, labels.unsqueeze(1)).squeeze()  # correct class scores
        max_scores = logits.max(dim=1)[0]  # maximum scores
        
        loss = F.relu(margin + max_scores - scores)
        return loss.mean() 