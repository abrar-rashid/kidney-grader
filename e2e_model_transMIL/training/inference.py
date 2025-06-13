# Credit to Claude for help with visualisation code

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import tempfile

from e2e_model_transMIL.models.transmil_regressor import TransMILRegressor
from e2e_model_transMIL.data.tubule_patch_extractor import TubulePatchExtractor
from e2e_model_transMIL.data.data_utils import visualize_attention_on_wsi, create_attention_summary_plot, extract_top_patches_for_visualization
from e2e_model_transMIL.data.data_utils import create_smooth_attention_heatmap, verify_coordinate_alignment
from e2e_model_transMIL.training.metrics import TubulitisMetrics


class TransMILInference:
    # comprehensive inference module with attention visualization
    
    def __init__(self, checkpoint_path: str, config: Dict, labels_csv_path: str = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # optionally load ground truth labels
        self.labels_df = None
        if labels_csv_path and Path(labels_csv_path).exists():
            self.labels_df = pd.read_csv(labels_csv_path)
            print(f"loaded ground truth labels from {labels_csv_path}")
        elif labels_csv_path:
            print(f"warning: labels file not found: {labels_csv_path}")
        
        # load model
        self.model = self._load_model(checkpoint_path)
        
        # setup patch extractor with full config
        self.patch_extractor = TubulePatchExtractor(
            config=config,
            patch_size=config['data']['patch_size'],
            overlap=config['data']['overlap'],
            tubule_class_id=config['data']['tubule_class_id'],
            min_tubule_ratio=config['data']['min_tubule_ratio']
        )
        
        print(f"inference model loaded on {self.device}")
        
    def _load_model(self, checkpoint_path: str) -> TransMILRegressor:
        # load trained model from checkpoint
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = TransMILRegressor(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"loaded model from {checkpoint_path}")
        print(f"training epoch: {checkpoint.get('epoch', 'unknown')}")
        
        return model
    
    def predict_single_wsi(self, wsi_path: str, 
                          output_dir: str = None,
                          save_visualizations: bool = True) -> Dict:
        
        wsi_path = Path(wsi_path)
        slide_name = wsi_path.stem
        
        if output_dir:
            output_dir = Path(output_dir)
            wsi_output_dir = output_dir / slide_name
            wsi_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            wsi_output_dir = None
            
        extraction_result = self.patch_extractor.extract_tubule_patches(
            str(wsi_path), 
            str(wsi_output_dir) if wsi_output_dir else None
        )
        
        if wsi_output_dir:
            segmentation_dir = wsi_output_dir / 'segmentation'
            if segmentation_dir.exists() and not any(segmentation_dir.iterdir()):
                try:
                    segmentation_dir.rmdir()
                    print(f"cleaned up empty segmentation directory")
                except OSError:
                    pass  
        
        features = torch.from_numpy(extraction_result['features']).float().unsqueeze(0)  # add batch dim
        coordinates = torch.from_numpy(extraction_result['normalized_coordinates']).float().unsqueeze(0)
        
        features = features.to(self.device)
        coordinates = coordinates.to(self.device)

        with torch.no_grad():
            results = self.model.predict(features, coordinates)
        
        predicted_score = results['predicted_scores'][0]
        attention_weights = results['attention_weights'][0].cpu().numpy()
        top_attention_indices = results['top_attention_indices'][0].cpu().numpy()
        top_attention_values = results['top_attention_values'][0].cpu().numpy()
        
        inference_results = {
            'slide_name': slide_name,
            'predicted_score': predicted_score,
            'num_patches': extraction_result['num_patches'],
            'attention_weights': attention_weights,
            'top_attention_indices': top_attention_indices,
            'top_attention_values': top_attention_values,
            'patch_coordinates': extraction_result['coordinates'],
            'normalized_coordinates': extraction_result['normalized_coordinates']
        }
        
        if wsi_output_dir:
            self._create_visualizations(
                wsi_path=wsi_path,
                inference_results=inference_results,
                extraction_result=extraction_result,
                output_dir=wsi_output_dir,
                save_visualizations=save_visualizations
            )
            
            self._save_inference_summary(inference_results, wsi_output_dir)
        
        print(f"predicted tubulitis score: {predicted_score:.3f}")
        print(f"top attention weight: {top_attention_values[0]:.4f}")
        print(f"results saved to: {wsi_output_dir}")
        
        return inference_results
    
    def _create_visualizations(self, wsi_path: Path, 
                              inference_results: Dict,
                              extraction_result: Dict,
                              output_dir: Path,
                              save_visualizations: bool = True):
        # create comprehensive visualizations - split into heavy and light
        
        slide_name = inference_results['slide_name']
        attention_weights = inference_results['attention_weights']
        patch_coordinates = inference_results['patch_coordinates']
        predicted_score = inference_results['predicted_score']
        
        # HEAVY VISUALIZATIONS - only if save_visualizations is True
        if save_visualizations:
            # 1. attention heatmap overlay on WSI
            heatmap_path = output_dir / "patch_attention_heatmap.png"
            visualize_attention_on_wsi(
                wsi_path=str(wsi_path),
                attention_weights=attention_weights,
                patch_coordinates=patch_coordinates,
                segmentation_mask=extraction_result['segmentation_mask'],
                patch_size=extraction_result['patch_size'],
                output_path=str(heatmap_path),
                top_k=self.config['visualization']['top_k_patches'],
                alpha=self.config['visualization']['heatmap_alpha']
            )
            
            # 1b. smooth weather-like attention heatmap
            smooth_heatmap_path = output_dir / "smooth_attention_heatmap.png"
            create_smooth_attention_heatmap(
                wsi_path=str(wsi_path),
                attention_weights=attention_weights,
                patch_coordinates=patch_coordinates,
                segmentation_mask=extraction_result['segmentation_mask'],
                patch_size=extraction_result['patch_size'],
                output_path=str(smooth_heatmap_path),
                alpha=self.config['visualization']['heatmap_alpha']
            )
        
        # LIGHT VISUALIZATIONS - always generate these
        # 2. attention summary plot
        summary_path = output_dir / "attention_summary.png"
        create_attention_summary_plot(
            attention_weights=attention_weights,
            patch_coordinates=patch_coordinates,
            slide_name=slide_name,
            predicted_score=predicted_score,
            true_score=np.nan,  # no ground truth in inference
            output_path=str(summary_path)
        )
        
        # 3. top attended patches visualization
        self._create_top_patches_visualization(
            inference_results=inference_results,
            extraction_result=extraction_result,
            output_path=output_dir / "top_patches.png"
        )
        
        # 4. attention distribution analysis
        self._create_attention_analysis(
            inference_results=inference_results,
            output_path=output_dir / "attention_analysis.png"
        )
        
        print(f"visualizations saved to {output_dir}")
        if save_visualizations:
            print("  heavy attention heatmaps included")
        else:
            print("  skipped heavy attention heatmaps (use --visualise to include)")
    
    def _create_top_patches_visualization(self, inference_results: Dict,
                                        extraction_result: Dict,
                                        output_path: Path):
        # visualize top-k attended patches with actual patch images
        
        top_indices = inference_results['top_attention_indices'][:10]  # top 10
        top_values = inference_results['top_attention_values'][:10]
        patch_coordinates = inference_results['patch_coordinates']
        
        # create figure
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"{inference_results['slide_name']}: Top Attended Patches", fontsize=14)
        
        # try to extract actual patch images
        try:
            # get WSI path from the slide name
            wsi_base_dir = Path("../all_wsis")
            slide_name = inference_results['slide_name']
            wsi_path = None
            
            for wsi_set_dir in wsi_base_dir.glob("wsi_set*"):
                possible_paths = [wsi_set_dir / slide_name, wsi_set_dir / f"{slide_name}.svs"]
                for path in possible_paths:
                    if path.exists():
                        wsi_path = path
                        break
                if wsi_path:
                    break
            
            if wsi_path and wsi_path.exists():
                # extract patches at top attention coordinates
                patch_images = self._extract_patches_at_coordinates(
                    str(wsi_path), 
                    patch_coordinates[top_indices],
                    patch_size=extraction_result['patch_size']
                )
            else:
                patch_images = None
                
        except Exception as e:
            print(f"Warning: Could not extract patch images: {e}")
            patch_images = None
        
        for i, (idx, weight) in enumerate(zip(top_indices, top_values)):
            row = i // 5
            col = i % 5
            
            if patch_images is not None and i < len(patch_images):
                # show actual patch image
                axes[row, col].imshow(patch_images[i])
                axes[row, col].set_title(f"Rank {i+1}: {weight:.4f}", fontsize=10)
            else:
                # fallback to text placeholder
                axes[row, col].text(0.5, 0.5, f"Patch {idx}\nWeight: {weight:.4f}", 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f"Rank {i+1}")
            
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _extract_patches_at_coordinates(self, wsi_path: str, coordinates: np.ndarray, 
                                       patch_size: int) -> List[np.ndarray]:
        # extract patch images at specific coordinates
        try:
            from openslide import OpenSlide
            
            slide = OpenSlide(wsi_path)
            patch_images = []
            
            for coord in coordinates:
                x, y = int(coord[0]), int(coord[1])
                
                # extract patch at level 0 (highest resolution)
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                patch = patch.convert('RGB')  # remove alpha channel
                patch_array = np.array(patch)
                
                patch_images.append(patch_array)
            
            slide.close()
            return patch_images
            
        except ImportError:
            print("OpenSlide not available for patch extraction")
            return None
        except Exception as e:
            print(f"Error extracting patches: {e}")
            return None
    
    def _create_attention_analysis(self, inference_results: Dict, output_path: Path):
        # create detailed attention analysis plot
        
        attention_weights = inference_results['attention_weights']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{inference_results['slide_name']}: Attention Analysis", fontsize=14)
        
        # attention weight distribution
        axes[0, 0].hist(attention_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('attention weight distribution')
        axes[0, 0].set_xlabel('attention weight')
        axes[0, 0].set_ylabel('frequency')
        axes[0, 0].axvline(attention_weights.mean(), color='red', linestyle='--', 
                          label=f'mean: {attention_weights.mean():.4f}')
        axes[0, 0].legend()
        
        # cumulative attention
        sorted_weights = np.sort(attention_weights)[::-1]
        cumulative_attention = np.cumsum(sorted_weights) / sorted_weights.sum()
        
        axes[0, 1].plot(range(len(cumulative_attention)), cumulative_attention, 'b-', linewidth=2)
        axes[0, 1].set_title('cumulative attention distribution')
        axes[0, 1].set_xlabel('patch rank')
        axes[0, 1].set_ylabel('cumulative attention')
        axes[0, 1].grid(True, alpha=0.3)
        
        # find how many patches contain 50%, 80%, 90% of attention
        for threshold in [0.5, 0.8, 0.9]:
            idx = np.where(cumulative_attention >= threshold)[0][0]
            axes[0, 1].axhline(threshold, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].axvline(idx, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].text(idx, threshold, f'{threshold*100}%\n@{idx}', 
                           ha='left', va='bottom', fontsize=8)
        
        # attention weight rank plot
        axes[1, 0].plot(range(len(sorted_weights)), sorted_weights, 'g-', linewidth=2)
        axes[1, 0].set_title('attention weights by rank')
        axes[1, 0].set_xlabel('patch rank')
        axes[1, 0].set_ylabel('attention weight')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # attention statistics
        stats_text = f"""
        Total patches: {len(attention_weights)}
        Mean attention: {attention_weights.mean():.6f}
        Std attention: {attention_weights.std():.6f}
        Max attention: {attention_weights.max():.6f}
        Min attention: {attention_weights.min():.6f}
        
        Top 1% contains: {cumulative_attention[len(attention_weights)//100]:.1%} of attention
        Top 5% contains: {cumulative_attention[len(attention_weights)//20]:.1%} of attention
        Top 10% contains: {cumulative_attention[len(attention_weights)//10]:.1%} of attention
        
        Predicted Score: {inference_results['predicted_score']:.3f}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('attention statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_ground_truth_score(self, slide_name: str) -> Optional[float]:
        # get ground truth tubulitis score for a slide
        
        if self.labels_df is None:
            return None
        
        # try both with and without .svs extension
        slide_names_to_try = [slide_name]
        if slide_name.endswith('.svs'):
            slide_names_to_try.append(slide_name[:-4])  # remove .svs
        else:
            slide_names_to_try.append(f"{slide_name}.svs")  # add .svs
        
        for name in slide_names_to_try:
            slide_row = self.labels_df[self.labels_df['filename'] == name]
            if not slide_row.empty:
                target_column = self.config.get('data', {}).get('target_column', 'T')
                score = slide_row[target_column].iloc[0]
                if pd.notna(score):
                    return float(score)
        
        return None
    
    def _save_inference_summary(self, inference_results: Dict, output_dir: Path):
        # save comprehensive inference summary to JSON
        
        slide_name = inference_results['slide_name']
        predicted_score = inference_results['predicted_score']
        num_patches = inference_results['num_patches']
        attention_weights = inference_results['attention_weights']
        top_attention_indices = inference_results['top_attention_indices']
        top_attention_values = inference_results['top_attention_values']
        patch_coordinates = inference_results['patch_coordinates']
        
        # get ground truth score if available
        ground_truth_score = self._get_ground_truth_score(slide_name)
        
        # prepare top patches info
        top_patches_info = []
        for rank, (idx, weight) in enumerate(zip(top_attention_indices[:10], top_attention_values[:10])):
            coord = patch_coordinates[idx]
            top_patches_info.append({
                'rank': rank + 1,
                'patch_index': int(idx),
                'coordinates': [int(coord[0]), int(coord[1])],
                'attention_weight': float(weight)
            })
        
        # create comprehensive summary
        slide_info = {
            'slide_name': slide_name,
            'predicted_tubulitis_score': float(predicted_score),
        }
        
        # add ground truth score directly below predicted score if available
        if ground_truth_score is not None:
            slide_info['ground_truth_tubulitis_score'] = float(ground_truth_score)
        
        slide_info.update({
            'number_of_tubule_patches': int(num_patches),
            'inference_timestamp': pd.Timestamp.now().isoformat()
        })
        
        summary = {
            'slide_info': slide_info,
            'attention_statistics': {
                'max_attention_weight': float(attention_weights.max()),
                'mean_attention_weight': float(attention_weights.mean()),
                'std_attention_weight': float(attention_weights.std()),
                'min_attention_weight': float(attention_weights.min()),
                'median_attention_weight': float(np.median(attention_weights))
            },
            'attention_distribution': {
                'top_1_percent_mean': float(np.sort(attention_weights)[-max(1, len(attention_weights)//100):].mean()),
                'top_5_percent_mean': float(np.sort(attention_weights)[-max(1, len(attention_weights)//20):].mean()),
                'top_10_percent_mean': float(np.sort(attention_weights)[-max(1, len(attention_weights)//10):].mean()),
                'bottom_50_percent_mean': float(np.sort(attention_weights)[:len(attention_weights)//2].mean())
            },
            'top_attended_patches': top_patches_info,
            'model_info': {
                'model_type': 'TransMIL',
                'feature_extraction': self.config.get('feature_extraction', {}).get('model', 'UNI'),
                'patch_size': self.config['data']['patch_size'],
                'min_tubule_ratio': self.config['data']['min_tubule_ratio']
            }
        }
        
        # save to JSON
        summary_path = output_dir / "inference_summary.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"inference summary saved to: {summary_path}")
    
    def predict_batch_wsis(self, wsi_paths: List[str], 
                          output_dir: str,
                          save_individual_visualizations: bool = True) -> pd.DataFrame:
        # batch prediction for multiple WSIs
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for wsi_path in wsi_paths:
            try:
                # run inference - predict_single_wsi will create individual folders automatically
                result = self.predict_single_wsi(
                    wsi_path=wsi_path,
                    output_dir=str(output_dir),  # pass main output dir, individual folders created automatically
                    save_visualizations=save_individual_visualizations
                )
                
                results.append({
                    'slide_name': result['slide_name'],
                    'predicted_score': result['predicted_score'],
                    'num_patches': result['num_patches'],
                    'max_attention': result['top_attention_values'][0],
                    'mean_attention': result['attention_weights'].mean(),
                    'std_attention': result['attention_weights'].std()
                })
                
            except Exception as e:
                print(f"error processing {wsi_path}: {e}")
                results.append({
                    'slide_name': Path(wsi_path).stem,
                    'predicted_score': np.nan,
                    'num_patches': 0,
                    'max_attention': np.nan,
                    'mean_attention': np.nan,
                    'std_attention': np.nan
                })
        
        # create summary dataframe
        results_df = pd.DataFrame(results)
        
        # save results
        results_csv_path = output_dir / "batch_predictions.csv"
        results_df.to_csv(results_csv_path, index=False)
        
        # create summary visualization
        self._create_batch_summary_visualization(results_df, output_dir)
        
        print(f"batch prediction completed. results saved to {results_csv_path}")
        
        return results_df
    
    def _create_batch_summary_visualization(self, results_df: pd.DataFrame, output_dir: Path):
        # create summary visualization for batch predictions
        
        valid_results = results_df.dropna()
        
        if len(valid_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('batch prediction summary', fontsize=14)
        
        # predicted score distribution
        axes[0, 0].hist(valid_results['predicted_score'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('predicted score distribution')
        axes[0, 0].set_xlabel('tubulitis score')
        axes[0, 0].set_ylabel('frequency')
        axes[0, 0].axvline(valid_results['predicted_score'].mean(), color='red', linestyle='--',
                          label=f'mean: {valid_results["predicted_score"].mean():.2f}')
        axes[0, 0].legend()
        
        # number of patches vs predicted score
        axes[0, 1].scatter(valid_results['num_patches'], valid_results['predicted_score'], alpha=0.6)
        axes[0, 1].set_title('patches vs predicted score')
        axes[0, 1].set_xlabel('number of patches')
        axes[0, 1].set_ylabel('predicted score')
        
        # attention statistics
        axes[1, 0].scatter(valid_results['mean_attention'], valid_results['predicted_score'], 
                          alpha=0.6, label='mean attention')
        axes[1, 0].scatter(valid_results['max_attention'], valid_results['predicted_score'], 
                          alpha=0.6, label='max attention')
        axes[1, 0].set_title('attention vs predicted score')
        axes[1, 0].set_xlabel('attention weight')
        axes[1, 0].set_ylabel('predicted score')
        axes[1, 0].legend()
        
        # summary statistics
        stats_text = f"""
        Total WSIs processed: {len(results_df)}
        Successful predictions: {len(valid_results)}
        
        Predicted Score Statistics:
        Mean: {valid_results['predicted_score'].mean():.3f}
        Std:  {valid_results['predicted_score'].std():.3f}
        Min:  {valid_results['predicted_score'].min():.3f}
        Max:  {valid_results['predicted_score'].max():.3f}
        
        Score Distribution:
        Score 0-1: {((valid_results['predicted_score'] >= 0) & (valid_results['predicted_score'] < 1)).sum()}
        Score 1-2: {((valid_results['predicted_score'] >= 1) & (valid_results['predicted_score'] < 2)).sum()}
        Score 2-3: {((valid_results['predicted_score'] >= 2) & (valid_results['predicted_score'] <= 3)).sum()}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('summary statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "batch_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    def verify_coordinate_alignment(self, wsi_path: str, output_dir: str = None):
        """
        run coordinate alignment verification for debugging visualization issues
        """
        
        if output_dir is None:
            output_dir = f"debug_alignment_{Path(wsi_path).stem}"
        
        print(f"running coordinate alignment verification for {Path(wsi_path).name}")
        
        # create temporary directory for patch extraction (segmentation needs an output path)
        with tempfile.TemporaryDirectory() as temp_dir:
            # extract patches and get inference results
            extraction_result = self.patch_extractor.extract_tubule_patches(str(wsi_path), temp_dir)
            
            if extraction_result['num_patches'] == 0:
                print("no patches extracted - cannot verify alignment")
                return
            
            # run inference to get attention weights
            features = torch.from_numpy(extraction_result['features']).float().unsqueeze(0)
            coordinates = torch.from_numpy(extraction_result['normalized_coordinates']).float().unsqueeze(0)
            
            features = features.to(self.device)
            coordinates = coordinates.to(self.device)
            
            with torch.no_grad():
                results = self.model.predict(features, coordinates)
            
            attention_weights = results['attention_weights'][0].cpu().numpy()
            
            # run verification (this will save to the permanent output_dir)
            verify_coordinate_alignment(
                wsi_path=str(wsi_path),
                attention_weights=attention_weights,
                patch_coordinates=extraction_result['coordinates'],
                segmentation_mask=extraction_result['segmentation_mask'],
                patch_size=extraction_result['patch_size'],
                output_dir=output_dir,
                top_k=10
            )
        
        print(f"coordinate alignment verification completed - results in {output_dir}")


def load_inference_model(checkpoint_path: str, config_path: str) -> TransMILInference:
    # convenience function to load inference model
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return TransMILInference(checkpoint_path, config) 