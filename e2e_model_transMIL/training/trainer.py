import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from e2e_model_transMIL.models.transmil_regressor import TransMILRegressor
from e2e_model_transMIL.data.data_utils import create_cv_splits, visualize_attention_on_wsi, create_attention_summary_plot
from e2e_model_transMIL.data.data_utils import create_smooth_attention_heatmap
from e2e_model_transMIL.data.tubule_dataset import create_dataloaders
from e2e_model_transMIL.training.metrics import TubulitisMetrics


class TransMILTrainer:
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        self.current_epoch = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        self._setup_directories()
        
    def _setup_directories(self):
        
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config['visualization']['save_attention_heatmaps']:
            self.viz_dir = Path(self.config['visualization']['attention_output_dir'])
            self.viz_dir.mkdir(parents=True, exist_ok=True)
            
    def _create_model(self) -> TransMILRegressor:
        model = TransMILRegressor(self.config)
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"model created: {total_params:,} total params, {trainable_params:,} trainable")
        
        return model
    
    def _create_optimizer(self, model: TransMILRegressor) -> optim.Optimizer:
        
        training_config = self.config['training']
        
        transformer_params = []
        regressor_params = []
        
        for name, param in model.named_parameters():
            if 'aggregator' in name:
                transformer_params.append(param)
            else:
                regressor_params.append(param)
        
        param_groups = [
            {'params': transformer_params, 'lr': training_config['lr'] * 0.5},
            {'params': regressor_params, 'lr': training_config['lr']}
        ]
        
        if training_config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                weight_decay=training_config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"optimizer {training_config['optimizer']} not supported")
            
        return optimizer
    
    def _create_scheduler(self, optimizer: optim.Optimizer, train_loader: DataLoader):
        
        training_config = self.config['training']
        total_steps = len(train_loader) * training_config['epochs']
        warmup_steps = len(train_loader) * training_config['warmup_epochs']
        
        if training_config['scheduler'] == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - warmup_steps,
                eta_min=training_config['lr'] * 0.01
            )
        else:
            scheduler = None
            
        return scheduler, warmup_steps
    
    def _warmup_scheduler(self, optimizer: optim.Optimizer, step: int, warmup_steps: int):
        if step < warmup_steps:
            warmup_factor = step / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
                
    def train_epoch(self, model: TransMILRegressor, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, scheduler=None) -> Dict[str, float]:
        
        model.train()
        epoch_losses = {
            'total_loss': [],
            'bag_loss': [],
            'instance_loss': [],
            'attention_reg': []
        }
        
        metrics = TubulitisMetrics()
        
        progress_bar = tqdm(train_loader, desc=f"epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            features = batch['features'].to(self.device)
            coordinates = batch['coordinates'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(features, coordinates, labels)
                    loss = outputs['loss']
                    
                self.scaler.scale(loss).backward()
                
                if self.config['training'].get('gradient_clip_value'):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['training']['gradient_clip_value']
                    )
                    
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(features, coordinates, labels)
                loss = outputs['loss']
                
                loss.backward()
                
                if self.config['training'].get('gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['training']['gradient_clip_value']
                    )
                    
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_losses['total_loss'].append(loss.item())
            epoch_losses['bag_loss'].append(outputs['bag_loss'].item())
            epoch_losses['instance_loss'].append(outputs['instance_loss'].item())
            epoch_losses['attention_reg'].append(outputs['attention_reg'].item())
            
            predictions = outputs['predictions']
            metrics.update(predictions, labels, batch.get('slide_names'))
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bag': f"{outputs['bag_loss'].item():.4f}",
                'inst': f"{outputs['instance_loss'].item():.4f}"
            })
        
        epoch_metrics = metrics.compute()
        
        for key in epoch_losses:
            epoch_metrics[f'train_{key}'] = np.mean(epoch_losses[key])
            
        return epoch_metrics
    
    def validate_epoch(self, model: TransMILRegressor, val_loader: DataLoader) -> Dict[str, float]:
        
        model.eval()
        val_losses = {
            'total_loss': [],
            'bag_loss': [],
            'instance_loss': [],
            'attention_reg': []
        }
        
        metrics = TubulitisMetrics()
        all_attention_weights = []
        all_slide_info = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="validation"):
                features = batch['features'].to(self.device)
                coordinates = batch['coordinates'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = model(features, coordinates, labels)
                else:
                    outputs = model(features, coordinates, labels)
                
                val_losses['total_loss'].append(outputs['loss'].item())
                val_losses['bag_loss'].append(outputs['bag_loss'].item())
                val_losses['instance_loss'].append(outputs['instance_loss'].item())
                val_losses['attention_reg'].append(outputs['attention_reg'].item())
                
                predictions = outputs['predictions']
                metrics.update(predictions, labels, batch.get('slide_names'))
                
                attention_weights = outputs['attention_weights'].cpu().numpy()
                all_attention_weights.append(attention_weights)
                
                if batch.get('slide_names'):
                    all_slide_info.append({
                        'slide_names': batch['slide_names'],
                        'predictions': predictions.cpu().numpy(),
                        'labels': labels.cpu().numpy(),
                        'coordinates': batch['coordinates'].cpu().numpy()
                    })
        
        val_metrics = metrics.compute()
        
        prefixed_val_metrics = {}
        for key, value in val_metrics.items():
            prefixed_val_metrics[f'val_{key}'] = value
        
        for key in val_losses:
            prefixed_val_metrics[f'val_{key}'] = np.mean(val_losses[key])
        
        if (self.config['visualization']['save_attention_heatmaps'] and 
            self.current_epoch % 10 == 0):
            self._save_attention_visualizations(all_attention_weights, all_slide_info)
        
        return prefixed_val_metrics
    
    def _save_attention_visualizations(self, attention_weights: List[np.ndarray], 
                                     slide_info: List[Dict]):
        
        for i, (attn_weights, info) in enumerate(zip(attention_weights[:5], slide_info[:5])):
            
            slide_name = info['slide_names'][0] if isinstance(info['slide_names'], list) else info['slide_names']
            prediction = info['predictions'][0]
            true_label = info['labels'][0]
            coordinates = info['coordinates'][0]
            
            summary_path = self.viz_dir / f"epoch_{self.current_epoch}_{slide_name}_summary.png"
            create_attention_summary_plot(
                attn_weights[0],
                coordinates,
                slide_name,
                prediction,
                true_label,
                str(summary_path)
            )
    
    def train_fold(self, fold_idx: int, train_loader: DataLoader, 
                  val_loader: DataLoader) -> Dict[str, float]:
        
        print(f"\ntraining fold {fold_idx}")
        print(f"train samples: {len(train_loader.dataset)}")
        print(f"val samples: {len(val_loader.dataset)}")
        
        model = self._create_model()
        optimizer = self._create_optimizer(model)
        scheduler, warmup_steps = self._create_scheduler(optimizer, train_loader)
        
        best_val_metric = float('inf')
        patience_counter = 0
        train_history = []
        val_history = []
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            if epoch < self.config['training']['warmup_epochs']:
                for batch_idx in range(len(train_loader)):
                    step = epoch * len(train_loader) + batch_idx
                    self._warmup_scheduler(optimizer, step, warmup_steps)
            
            train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler)
            train_history.append(train_metrics)
            
            val_metrics = self.validate_epoch(model, val_loader)
            val_history.append(val_metrics)
            
            monitor_metric = val_metrics[self.config['validation']['monitor']]
            
            if monitor_metric < best_val_metric:
                best_val_metric = monitor_metric
                patience_counter = 0
                
                self._save_checkpoint(model, optimizer, epoch, fold_idx, is_best=True)
                
            else:
                patience_counter += 1
            
            if epoch % self.config['logging']['log_interval'] == 0:
                print(f"epoch {epoch}: train_loss={train_metrics['train_total_loss']:.4f}, "
                      f"val_loss={val_metrics['val_total_loss']:.4f}, "
                      f"val_mse={val_metrics['val_mse']:.4f}, "
                      f"patience={patience_counter}/{self.config['training']['early_stopping_patience']}")
            
            if patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"early stopping at epoch {epoch}")
                break
                
            if epoch % self.config['checkpoint']['save_every_n_epochs'] == 0:
                self._save_checkpoint(model, optimizer, epoch, fold_idx, is_best=False)
        
        best_checkpoint_path = self.checkpoint_dir / f"fold_{fold_idx}_best.pth"
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_val_metrics = self.validate_epoch(model, val_loader)
        
        return final_val_metrics
    
    def _save_checkpoint(self, model: TransMILRegressor, optimizer: optim.Optimizer, 
                        epoch: int, fold_idx: int, is_best: bool = False):
        
        checkpoint = {
            'epoch': epoch,
            'fold': fold_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'best_val_metric': self.best_val_metric
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"fold_{fold_idx}_best.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"fold_{fold_idx}_epoch_{epoch}.pth"
            
        torch.save(checkpoint, checkpoint_path)
        
    def cross_validate(self) -> Dict[str, Dict]:
        
        labels_df = pd.read_csv(self.config['data']['labels_file'])
        cv_splits = create_cv_splits(
            labels_df,
            target_column=self.config['data']['target_column'],
            n_folds=self.config['data']['n_folds'],
            test_size=self.config['data']['test_size'],
            random_seed=self.config['data']['random_seed']
        )
        
        fold_results = {}
        
        for fold_idx in range(self.config['data']['n_folds']):
            print(f"\n{'='*50}")
            print(f"fold {fold_idx + 1}/{self.config['data']['n_folds']}")
            print(f"{'='*50}")
            
            fold_splits = cv_splits[fold_idx]
            
            train_loader, val_loader, _ = create_dataloaders(
                train_slides=fold_splits['train'],
                val_slides=fold_splits['val'],
                test_slides=fold_splits['test'],
                labels_df=labels_df,
                features_dir=self.config['data']['features_dir'],
                config=self.config
            )
            
            fold_metrics = self.train_fold(fold_idx, train_loader, val_loader)
            fold_results[fold_idx] = fold_metrics
        
        cv_summary = self._summarize_cv_results(fold_results)
        
        results_path = self.checkpoint_dir / "cv_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump({
                'fold_results': fold_results,
                'cv_summary': cv_summary
            }, f, default_flow_style=False)
        
        print(f"\ncross-validation completed. results saved to {results_path}")
        
        return {
            'fold_results': fold_results,
            'cv_summary': cv_summary
        }
    
    def _summarize_cv_results(self, fold_results: Dict[int, Dict]) -> Dict[str, float]:
        
        summary = {}
        
        metric_names = list(fold_results[0].keys())
        
        for metric_name in metric_names:
            values = [fold_results[fold][metric_name] for fold in fold_results]
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
        
        print(f"\ncross-validation summary:")
        print(f"val_mse: {summary['val_mse_mean']:.4f} ± {summary['val_mse_std']:.4f}")
        print(f"val_mae: {summary['val_mae_mean']:.4f} ± {summary['val_mae_std']:.4f}")
        
        if 'val_discrete_accuracy_mean' in summary:
            print(f"discrete_accuracy: {summary['val_discrete_accuracy_mean']:.4f} ± {summary['val_discrete_accuracy_std']:.4f}")
        
        return summary