import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import wandb
from tqdm import tqdm
from rich.console import Console
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from models import create_clam_regressor
from .dataset import create_data_loaders

console = Console()


class CLAMTrainer:
    def __init__(self, config: Dict[str, Any], model_save_dir: str):
        self.config = config
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(config['hardware']['device'])
        console.print(f"Using device: {self.device}")
        
        # for reproducibility
        self._set_random_seeds(config['random_seed'])
        
        self.model = create_clam_regressor(config).to(self.device)
        console.print(f"Created CLAM model with {self._count_parameters()} parameters")
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.use_amp = config['hardware']['mixed_precision']
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            console.print("Using automatic mixed precision")
        
        self.current_epoch = 0
        self.best_metric = float('inf')  # For MSE, lower is better
        self.train_history = []
        self.val_history = []
        
        self.patience = config['training']['early_stopping_patience']
        self.patience_counter = 0
        
        self.use_wandb = config['logging']['use_wandb']
        if self.use_wandb:
            self._init_wandb()
    
    def _set_random_seeds(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _create_optimizer(self):
        training_config = self.config['training']
        
        if training_config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=training_config['lr'],
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=training_config['lr'],
                weight_decay=training_config['weight_decay'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
        
        return optimizer
    
    def _create_scheduler(self):
        training_config = self.config['training']
        
        if training_config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs']
            )
        elif training_config['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
        elif training_config['scheduler'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=20
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _init_wandb(self):
        wandb.init(
            project=self.config['logging']['project_name'],
            config=self.config,
            name=f"clam_tubulitis_{self.config['random_seed']}"
        )
        wandb.watch(self.model, log="all", log_freq=100)
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        # Regression metrics. See report for more details
        # Ensure predictions are in valid range [0, 3]
        predictions = np.clip(predictions, 0.0, 3.0)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        r2 = r2_score(targets, predictions)
        
        # pearson correlation, an important metric for this problem
        try:
            pearson_r, pearson_p = pearsonr(targets, predictions)
        except:
            pearson_r, pearson_p = 0.0, 1.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_bag_loss = 0.0
        total_instance_loss = 0.0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} - Training")
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    results = self.model(features, labels)
                    loss = results['loss']
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(features, labels)
                loss = results['loss']
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_bag_loss += results['bag_loss'].item()
            total_instance_loss += results['instance_loss'].item()
            
            with torch.no_grad():
                preds = results['logits'].squeeze().cpu().numpy()
                targs = labels.cpu().numpy()
                
                if len(preds.shape) == 0:
                    preds = [preds.item()]
                    targs = [targs.item()]
                
                predictions.extend(preds)
                targets.extend(targs)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bag_loss': f"{results['bag_loss'].item():.4f}",
                'inst_loss': f"{results['instance_loss'].item():.4f}"
            })
            
            if self.use_wandb and batch_idx % self.config['logging']['log_interval'] == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_bag_loss': results['bag_loss'].item(),
                    'train/batch_instance_loss': results['instance_loss'].item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_bag_loss = total_bag_loss / len(train_loader)
        avg_instance_loss = total_instance_loss / len(train_loader)
        
        metrics = self.calculate_metrics(np.array(predictions), np.array(targets))
        
        epoch_results = {
            'loss': avg_loss,
            'bag_loss': avg_bag_loss,
            'instance_loss': avg_instance_loss,
            **{f"train_{k}": v for k, v in metrics.items()}
        }
        
        return epoch_results
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        total_bag_loss = 0.0
        total_instance_loss = 0.0
        predictions = []
        targets = []
        slide_names = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} - Validation")
            
            for batch in pbar:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                results = self.model(features, labels)
                
                total_loss += results['loss'].item()
                total_bag_loss += results['bag_loss'].item()
                total_instance_loss += results['instance_loss'].item()
                
                preds = results['logits'].squeeze().cpu().numpy()
                targs = labels.cpu().numpy()
                
                if len(preds.shape) == 0:  # Single sample
                    preds = [preds.item()]
                    targs = [targs.item()]
                
                predictions.extend(preds)
                targets.extend(targs)
                slide_names.extend(batch['slide_names'])
                
                pbar.set_postfix({'val_loss': f"{results['loss'].item():.4f}"})
        
        avg_loss = total_loss / len(val_loader)
        avg_bag_loss = total_bag_loss / len(val_loader)
        avg_instance_loss = total_instance_loss / len(val_loader)
        
        metrics = self.calculate_metrics(np.array(predictions), np.array(targets))
        
        epoch_results = {
            'val_loss': avg_loss,
            'val_bag_loss': avg_bag_loss,
            'val_instance_loss': avg_instance_loss,
            **{f"val_{k}": v for k, v in metrics.items()}
        }
        
        val_predictions = pd.DataFrame({
            'slide_name': slide_names,
            'prediction': predictions,
            'target': targets,
            'epoch': self.current_epoch
        })
        
        return epoch_results, val_predictions
    
    def save_checkpoint(self, epoch_results: Dict[str, float], is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'epoch_results': epoch_results
        }
        
        checkpoint_path = self.model_save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.model_save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            console.print(f"[green]New best model saved! Val MSE: {epoch_results['val_mse']:.4f}[/green]")
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        # main training loop
        console.print(f"\n[bold blue]Starting training for {self.config['training']['epochs']} epochs[/bold blue]")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            train_results = self.train_epoch(train_loader)
            self.train_history.append(train_results)
            
            val_results, val_predictions = self.validate_epoch(val_loader)
            self.val_history.append(val_results)
            
            epoch_results = {**train_results, **val_results}
            
            # update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['val_loss'])
                else:
                    self.scheduler.step()
            
            current_metric = val_results['val_mse']
            is_best = current_metric < self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch_results, is_best)
            
            if self.use_wandb:
                wandb.log(epoch_results)
            
            console.print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}:")
            console.print(f"  Train Loss: {train_results['loss']:.4f} | Val Loss: {val_results['val_loss']:.4f}")
            console.print(f"  Train MSE: {train_results['train_mse']:.4f} | Val MSE: {val_results['val_mse']:.4f}")
            console.print(f"  Train R²: {train_results['train_r2']:.4f} | Val R²: {val_results['val_r2']:.4f}")
            console.print(f"  Best Val MSE: {self.best_metric:.4f} | Patience: {self.patience_counter}/{self.patience}")
            
            pred_dir = self.model_save_dir / "predictions"
            pred_dir.mkdir(exist_ok=True)
            val_predictions.to_csv(pred_dir / f"val_predictions_epoch_{epoch+1}.csv", index=False)
            
            if self.patience_counter >= self.patience:
                console.print(f"[yellow]Early stopping triggered after {epoch+1} epochs[/yellow]")
                break
        
        history_df = pd.DataFrame(self.train_history + self.val_history)
        history_df.to_csv(self.model_save_dir / "training_history.csv", index=False)
        
        console.print(f"\n[green]Training completed! Best validation MSE: {self.best_metric:.4f}[/green]")
        
        return {
            'best_metric': self.best_metric,
            'total_epochs': self.current_epoch + 1,
            'train_history': self.train_history,
            'val_history': self.val_history
        }


def main():
    parser = argparse.ArgumentParser(description="Train CLAM model for tubulitis scoring")
    parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    console.print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    features_dir = config['data']['features_dir']
    labels_file = config['data']['labels_file']
    split_file = f"{features_dir}/splits/data_splits.json"
    
    console.print("\n[bold blue]Creating data loaders...[/bold blue]")
    train_loader, val_loader, test_loader = create_data_loaders(
        features_dir, labels_file, split_file, config
    )
    
    model_save_dir = config['checkpoint']['save_dir']
    trainer = CLAMTrainer(config, model_save_dir)
    
    if args.resume:
        console.print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_metric = checkpoint['best_metric']
    
    training_results = trainer.train(train_loader, val_loader)
    
    results_path = Path(model_save_dir) / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    console.print(f"\n[bold green]Training pipeline completed![/bold green]")
    console.print(f"Results saved to: {model_save_dir}")


if __name__ == "__main__":
    main() 