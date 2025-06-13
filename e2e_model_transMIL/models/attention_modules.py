import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional


class PositionalEncoding2D(nn.Module):
    # 2D positional encoding for spatial relationships in WSI patches
    
    def __init__(self, d_model: int, max_patches: int = 10000):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.max_patches = max_patches
        
        # learnable 2D position embeddings
        self.row_embed = nn.Embedding(int(math.sqrt(max_patches)) + 1, d_model // 2)
        self.col_embed = nn.Embedding(int(math.sqrt(max_patches)) + 1, d_model // 2)
        
    def forward(self, patch_coords: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, _ = patch_coords.shape
        
        max_coord = patch_coords.max().item()
        grid_size = int(math.sqrt(num_patches)) + 1
        
        row_indices = (patch_coords[:, :, 1] / max_coord * grid_size).long()
        col_indices = (patch_coords[:, :, 0] / max_coord * grid_size).long()
        
        row_indices = torch.clamp(row_indices, 0, self.row_embed.num_embeddings - 1)
        col_indices = torch.clamp(col_indices, 0, self.col_embed.num_embeddings - 1)
        
        row_embeds = self.row_embed(row_indices)
        col_embeds = self.col_embed(col_indices)
        
        pos_embeds = torch.cat([row_embeds, col_embeds], dim=-1)
        return pos_embeds


class MultiHeadAttention(nn.Module):
    # multi-head attention with bidirectional processing for tubule relationship modelling
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        output = self.layer_norm(output + query)
        
        # return averaged attention weights across heads for visualisation
        attention_vis = attention_weights.mean(dim=1)
        
        return output, attention_vis


class BidirectionalTransformerBlock(nn.Module):
    # bidirectional transformer block for enhanced tubule pattern recognition
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super(BidirectionalTransformerBlock, self).__init__()
        
        self.forward_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.backward_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, pos_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pos = x + pos_encoding
        
        forward_out, forward_attn = self.forward_attention(x_pos, x_pos, x_pos)
        
        x_reversed = torch.flip(x_pos, dims=[1])
        backward_out, backward_attn = self.backward_attention(x_reversed, x_reversed, x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])
        backward_attn = torch.flip(backward_attn, dims=[2])
        
        attention_out = self.gate * forward_out + (1 - self.gate) * backward_out
        combined_attn = (forward_attn + backward_attn) / 2
        
        ffn_out = self.ffn(attention_out)
        output = self.layer_norm(ffn_out + attention_out)
        
        return output, combined_attn


class TransMILAggregator(nn.Module):
    # main aggregator for tubule-focused multiple instance learning
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, 
                 num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super(TransMILAggregator, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding2D(hidden_dim)
        
        self.transformer_blocks = nn.ModuleList([
            BidirectionalTransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # class token for bag-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # instance-level classifier for pseudo-labeling
        self.instance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # binary: relevant/irrelevant tubule patch
        )
        
    def forward(self, features: torch.Tensor, coordinates: torch.Tensor) -> dict:
        
        batch_size, num_patches, _ = features.shape
        
        # project to hidden dimension
        x = self.input_projection(features)
        
        # add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        cls_coords = torch.zeros(batch_size, 1, 2, device=coordinates.device)
        all_coords = torch.cat([cls_coords, coordinates], dim=1)
        
        pos_encoding = self.pos_encoding(all_coords)
        
        attention_maps = []
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x, pos_encoding)
            attention_maps.append(attention_weights)
        
        cls_features = x[:, 0] 
        patch_features = x[:, 1:]
        
        # instance-level predictions for pseudo-labeling
        instance_logits = self.instance_classifier(patch_features)
        
        # final attention weights (from last layer, averaged over heads)
        final_attention = attention_maps[-1][:, 0, 1:]  # attention from cls to patches
        
        return {
            'bag_features': cls_features,
            'patch_features': patch_features,
            'instance_logits': instance_logits,
            'attention_weights': final_attention,
            'attention_maps': attention_maps
        } 