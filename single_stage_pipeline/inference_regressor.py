
# from the kidney-grader root directory:
# python single_stage_pipeline/inference_regressor.py --wsi path/to/slide.svs --output_dir results/ --model single_stage_pipeline/checkpoints_regressor/cv_fold_0/best_model.pth

# run from the single_stage_pipeline directory:
# python inference_regressor.py --wsi path/to/slide.svs --output_dir results/ --model checkpoints_regressor/cv_fold_0/best_model.pth

# force regeneration of features:
# python single_stage_pipeline/inference_regressor.py --wsi path/to/slide.svs --output_dir results/ --model single_stage_pipeline/checkpoints_regressor/cv_fold_0/best_model.pth --force


import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
import time
from PIL import Image, ImageDraw, ImageFont
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cv2
import openslide

script_dir = Path(__file__).parent
root_dir = script_dir.parent 

sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(script_dir))

try:
    from single_stage_pipeline.models import create_clam_regressor
    from single_stage_pipeline.training.dataset import WSIFeaturesDataset
except ImportError:
    from models import create_clam_regressor
    from training.dataset import WSIFeaturesDataset

try:
    from single_stage_pipeline.preprocessing import UNIFeatureExtractor, KidneyPatchExtractor
except ImportError:
    try:
        from preprocessing import UNIFeatureExtractor, KidneyPatchExtractor
    except ImportError:
        try:
            from uni_feature_extractor import UNIFeatureExtractor
        except ImportError:
            from single_stage_pipeline.preprocessing.uni_feature_extractor import UNIFeatureExtractor
        
        try:
            from single_stage_pipeline.preprocessing.patch_extractor import KidneyPatchExtractor
        except ImportError:
            from preprocessing.patch_extractor import KidneyPatchExtractor


class CLAMRegressorInference:    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        
        self.load_model_and_config()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded CLAM regressor from: {model_path}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.count_parameters():,}")
    
    def load_model_and_config(self):
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            print("Loaded configuration from checkpoint")
        elif self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Loaded configuration from: {self.config_path}")
        else:
            # default configuration for inference
            self.config = self._get_default_config()
            print("Using default configuration")
        
        # Create model
        self.model = create_clam_regressor(self.config)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint.get('epoch', 'unknown')
            self.best_metric = checkpoint.get('best_metric', 'unknown')
            print(f"Model from epoch {self.epoch}, best validation MSE: {self.best_metric}")
        else:
            self.model.load_state_dict(checkpoint)
        
        # Check if this is a tubule-focused model
        self.is_tubule_focused = checkpoint.get('tubule_focused', False)
        self.masks_dir = checkpoint.get('masks_dir', None)
        
        if self.is_tubule_focused:
            print(f"Tubule-focused model detected")
            if self.masks_dir:
                print(f"Masks directory: {self.masks_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'model': {
                'name': 'CLAM',
                'task': 'regression',
                'clam': {
                    'gate': True,
                    'size_arg': 'small',
                    'dropout': 0.25,
                    'k_sample': 8,
                    'instance_loss_fn': 'svm'
                },
                'feature_dim': 1024,
                'hidden_dim': 256,
                'num_classes': 1
            }
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def predict_from_wsi(self, wsi_path: str, features_dir: str = None, 
                        extract_features: bool = True, force: bool = False) -> Dict[str, Any]:
        # force param is whether to force regeneration of patches, features, and visualizations
        wsi_path = Path(wsi_path)
        slide_name = wsi_path.stem
        
        print(f"\nProcessing WSI: {slide_name}")
        
        if extract_features:
            print("Extracting features...")
            features_path = self._extract_features(wsi_path, features_dir, force)
        else:
            if features_dir:
                features_path = Path(features_dir) / f"{slide_name}.h5"
            else:
                features_path = wsi_path.parent / f"{slide_name}.h5"
            
            if not features_path.exists():
                print(f"Features not found at {features_path}")
                print("Try setting extract_features=True to extract features")
                return None
        
        print("Loading features...")
        features, coordinates = self._load_features(features_path)
        
        # if self.is_tubule_focused and self.masks_dir:
        #     print("Applying tubule filtering...")
        #     features, coordinates = self._apply_tubule_filtering(
        #         features, coordinates, slide_name
        #     )
        
        print("Running inference...")
        start_time = time.time()
        
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            
            # forward pass
            results = self.model(features_tensor)
            
            prediction = results['logits'].squeeze().cpu().numpy()
            attention_weights = results['attention_weights'].squeeze().cpu().numpy()
        
        inference_time = time.time() - start_time
        
        prediction_score = float(prediction)
        
        # convert attention weights to interpretable format
        top_patches_idx = np.argsort(attention_weights)[-10:][::-1]  # top 10 patches used for bag-level prediction
        
        results = {
            'slide_name': slide_name,
            'prediction': prediction_score,
            'rounded_prediction': round(prediction_score),
            'tubulitis_grade': self._score_to_grade(prediction_score),
            'num_patches': len(features),
            'num_patches_used': len(features) if not self.is_tubule_focused else len(features),
            'inference_time': inference_time,
            'attention_weights': attention_weights,
            'top_patches': {
                'indices': top_patches_idx.tolist(),
                'coordinates': coordinates[top_patches_idx].tolist() if len(coordinates) > 0 else [],
                'attention_scores': attention_weights[top_patches_idx].tolist()
            },
            'model_info': {
                'is_tubule_focused': self.is_tubule_focused,
                'epoch': self.epoch,
                'best_metric': self.best_metric
            }
        }
        
        self._print_results(results)
        return results
    
    def _extract_features(self, wsi_path: Path, features_dir: str = None, force: bool = False) -> Path:
        # extract features from WSI using UNI foundation model
        if features_dir is None:
            features_dir = wsi_path.parent / "features"
        
        features_dir = Path(features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)
        
        features_path = features_dir / f"{wsi_path.stem}.h5"
        
        if features_path.exists() and not force:
            print(f"Features already exist: {features_path}")
            print(f"Use --force to regenerate features")
            return features_path
        elif features_path.exists() and force:
            print(f"Force regeneration enabled - removing existing features: {features_path}")
            features_path.unlink()
        
        config_path = Path(__file__).parent / "configs" / "patch_extraction.yaml"
        
        try:
            with open(config_path, 'r') as f:
                extraction_config = yaml.safe_load(f)
            
            print(f"Loaded config from: {config_path}")
            
            print(f"Extracting patches from WSI...")
            patch_extractor = KidneyPatchExtractor(extraction_config)
            patch_data = patch_extractor.extract_patches_from_wsi(str(wsi_path))
            
            if len(patch_data['patches']) == 0:
                raise ValueError("No patches extracted from WSI")
            
            print(f"Extracted {len(patch_data['patches'])} patches")
            print(f"Extracting features using UNI model...")
            
            feature_extractor = UNIFeatureExtractor(extraction_config)
            features = feature_extractor.extract_features_from_patches(patch_data['patches'])
            
            print(f"  Extracted features with shape: {features.shape}")
            
            with h5py.File(features_path, 'w') as f:
                f.create_dataset('features', data=features)
                f.create_dataset('coordinates', data=patch_data['coordinates'])
                f.attrs['slide_name'] = wsi_path.stem
                f.attrs['num_patches'] = len(features)
                f.attrs['feature_dim'] = features.shape[1]
                f.attrs['patch_size'] = 512
            
            print(f"Saved features to: {features_path}")
            return features_path
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            print(f"You may need to extract features separately using the preprocessing pipeline")
            raise e
    
    def _load_features(self, features_path: Path) -> tuple:
        # load features and coords from H5 file
        with h5py.File(features_path, 'r') as f:
            features = f['features'][:]
            coordinates = f['coordinates'][:] if 'coordinates' in f else np.array([])
        
        print(f"Loaded {len(features)} patches")
        return features, coordinates
    
    # def _apply_tubule_filtering(self, features: np.ndarray, coordinates: np.ndarray, 
    #                           slide_name: str) -> tuple:
    #     #Apply tubule filtering to patches
    #     if len(coordinates) == 0:
    #         print( No coordinates available for tubule filtering")
    #         return features, coordinates
        
    #     mask_path = Path(self.masks_dir) / slide_name / f"{slide_name}_tubule_mask.tiff"
        
    #     if not mask_path.exists():
    #         print(f"Tubule mask not found: {mask_path}")
    #         print("   Using all patches without filtering")
    #         return features, coordinates
        
    #     from PIL import Image
    #     import numpy as np
        
    #     tubule_mask = np.array(Image.open(mask_path))
        
    #     patch_size = 512
    #     overlap_threshold = 0.3
        
    #     valid_indices = []
    #     for i, (x, y) in enumerate(coordinates):
    #         x_start, y_start = int(x), int(y)
    #         x_end = min(x_start + patch_size, tubule_mask.shape[1])
    #         y_end = min(y_start + patch_size, tubule_mask.shape[0])
            
    #         if x_end > x_start and y_end > y_start:
    #             patch_mask = tubule_mask[y_start:y_end, x_start:x_end]
    #             tubule_pixels = np.sum(patch_mask > 0)
    #             total_pixels = patch_mask.size
                
    #             if total_pixels > 0:
    #                 overlap = tubule_pixels / total_pixels
    #                 if overlap >= overlap_threshold:
    #                     valid_indices.append(i)
        
    #     valid_indices = np.array(valid_indices)
        
    #     if len(valid_indices) > 0:
    #         filtered_features = features[valid_indices]
    #         filtered_coordinates = coordinates[valid_indices]
    #         print(f"Filtered to {len(filtered_features)} tubule-rich patches ({len(filtered_features)/len(features)*100:.1f}%)")
    #         return filtered_features, filtered_coordinates
    #     else:
    #         print("No patches met tubule overlap threshold, using all patches")
    #         return features, coordinates
    
    def _score_to_grade(self, score: float) -> str:
        # Convert continuous score to tubulitis grade
        if score < 0.5:
            return "T0 (No tubulitis)"
        elif score < 1.5:
            return "T1 (Mild tubulitis)"
        elif score < 2.5:
            return "T2 (Moderate tubulitis)"
        else:
            return "T3 (Severe tubulitis)"
    
    def _print_results(self, results: Dict[str, Any]):
        print(f"\nInference Results:")
        print(f"Predicted Score: {results['prediction']:.3f}")
        print(f"Rounded Score: {results['rounded_prediction']}")
        print(f"Tubulitis Grade: {results['tubulitis_grade']}")
        print(f"Patches Used: {results['num_patches']}")
        print(f"Inference Time: {results['inference_time']:.2f}s")
        
        # if results['model_info']['is_tubule_focused']:
        #     print(f"Tubule-focused model used")
        
        print(f"\nTop 5 Most Attended Patches:")
        for i, (idx, coord, attn) in enumerate(zip(
            results['top_patches']['indices'][:5],
            results['top_patches']['coordinates'][:5],
            results['top_patches']['attention_scores'][:5]
        )):
            if len(coord) >= 2:
                print(f"    {i+1}. Patch {idx} at ({coord[0]}, {coord[1]}) - Attention: {attn:.4f}")
            else:
                print(f"    {i+1}. Patch {idx} - Attention: {attn:.4f}")

    def create_attention_visualization(self, wsi_path: str, results: Dict[str, Any], 
                                     output_dir: Path) -> List[str]:
        #create attention visualization overlays on WSI thumbnail
        try:
            print(f"\nCreating attention visualizations...")
            
            slide = openslide.OpenSlide(wsi_path)
            
            # reasonably sized thumbnail (max 2000px on longest side)
            level_dims = slide.level_dimensions
            width, height = level_dims[0]
            
            max_thumb_size = 2000
            if max(width, height) > max_thumb_size:
                scale_factor = max_thumb_size / max(width, height)
                thumb_width = int(width * scale_factor)
                thumb_height = int(height * scale_factor)
            else:
                thumb_width, thumb_height = width, height
                scale_factor = 1.0
            
            print(f"Creating thumbnail: {thumb_width}x{thumb_height} (scale: {scale_factor:.4f})")
            thumbnail = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
            thumbnail_np = np.array(thumbnail)
            
            attention_weights = np.array(results['attention_weights'])
            coordinates = np.array(results['top_patches']['coordinates'])
            
            if len(coordinates) == 0:
                print("  No coordinates available for visualization")
                return []
            
            # create attention heatmap overlay
            patch_size_scaled = int(512 * scale_factor)  # Scale patch size to thumbnail            
            heatmap = np.zeros((thumb_height, thumb_width), dtype=np.float32)
            
            # map patch coordinates to thumbnail and overlay attention
            for i, (x, y) in enumerate(coordinates):
                if i >= len(attention_weights):
                    break
                    
                thumb_x = int(x * scale_factor)
                thumb_y = int(y * scale_factor)
                
                x_start = max(0, thumb_x)
                y_start = max(0, thumb_y)
                x_end = min(thumb_width, thumb_x + patch_size_scaled)
                y_end = min(thumb_height, thumb_y + patch_size_scaled)
                
                if x_end > x_start and y_end > y_start:
                    heatmap[y_start:y_end, x_start:x_end] = np.maximum(
                        heatmap[y_start:y_end, x_start:x_end], 
                        attention_weights[i]
                    )
            
            # normlalise
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            saved_files = []
            
            plt.figure(figsize=(15, 12))
            plt.imshow(thumbnail_np)
            plt.imshow(heatmap, cmap='jet', alpha=0.4, vmin=0, vmax=1)
            plt.colorbar(label='Attention Weight', shrink=0.8)
            plt.title(f'Attention Heatmap: {results["slide_name"]}\n'
                     f'Predicted Score: {results["prediction"]:.3f} ({results["tubulitis_grade"]})', 
                     fontsize=14, pad=20)
            plt.axis('off')
            
            heatmap_path = output_dir / f"{results['slide_name']}_attention_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_files.append(str(heatmap_path))
            print(f" Saved heatmap: {heatmap_path}")
            
            thumbnail_pil = Image.fromarray(thumbnail_np)
            draw = ImageDraw.Draw(thumbnail_pil)
            
            # get top 10 patches
            top_n = min(10, len(results['top_patches']['indices']))
            top_coords = results['top_patches']['coordinates'][:top_n]
            top_attentions = results['top_patches']['attention_scores'][:top_n]
            
            norm = Normalize(vmin=min(top_attentions), vmax=max(top_attentions))
            colormap = cm.get_cmap('YlOrRd')
            
            for i, ((x, y), attention) in enumerate(zip(top_coords, top_attentions)):
                thumb_x = int(x * scale_factor)
                thumb_y = int(y * scale_factor)
                
                color_rgba = colormap(norm(attention))
                color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
                
                x_end = min(thumb_width, thumb_x + patch_size_scaled)
                y_end = min(thumb_height, thumb_y + patch_size_scaled)
                
                draw.rectangle([thumb_x, thumb_y, x_end, y_end], 
                             outline=color_rgb, width=3)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                text = str(i + 1)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_bg = [thumb_x + 5, thumb_y + 5, 
                          thumb_x + text_width + 10, thumb_y + text_height + 10]
                draw.rectangle(text_bg, fill=(255, 255, 255, 180))
                draw.text((thumb_x + 8, thumb_y + 8), text, fill=color_rgb, font=font)
            
            top_patches_path = output_dir / f"{results['slide_name']}_top_patches.png"
            thumbnail_pil.save(top_patches_path, dpi=(300, 300))
            saved_files.append(str(top_patches_path))
            print(f"  Saved top patches: {top_patches_path}")
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(attention_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Attention Weight')
            plt.ylabel('Number of Patches')
            plt.title('Attention Weight Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            top_20_idx = np.argsort(attention_weights)[-20:][::-1]
            top_20_scores = attention_weights[top_20_idx]
            plt.bar(range(len(top_20_scores)), top_20_scores, color='coral')
            plt.xlabel('Patch Rank')
            plt.ylabel('Attention Weight')
            plt.title('Top 20 Attention Scores')
            plt.xticks(range(0, len(top_20_scores), 2))
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.text(0.1, 0.8, f"Slide: {results['slide_name']}", fontsize=12, weight='bold')
            plt.text(0.1, 0.7, f"Predicted Score: {results['prediction']:.3f}", fontsize=12)
            plt.text(0.1, 0.6, f"Tubulitis Grade: {results['tubulitis_grade']}", fontsize=12)
            plt.text(0.1, 0.5, f"Total Patches: {results['num_patches']}", fontsize=12)
            plt.text(0.1, 0.4, f"Max Attention: {attention_weights.max():.4f}", fontsize=12)
            plt.text(0.1, 0.3, f"Mean Attention: {attention_weights.mean():.4f}", fontsize=12)
            plt.text(0.1, 0.2, f"Inference Time: {results['inference_time']:.2f}s", fontsize=12)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title('Prediction Summary')
            
            plt.subplot(2, 2, 4)
            if len(coordinates) > 0:
                scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                                    c=attention_weights, cmap='YlOrRd', 
                                    s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, label='Attention Weight')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.title('Spatial Attention Distribution')
                plt.gca().invert_yaxis()
            
            plt.tight_layout()
            stats_path = output_dir / f"{results['slide_name']}_attention_stats.png"
            plt.savefig(stats_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_files.append(str(stats_path))
            print(f"Saved statistics: {stats_path}")
            
            slide.close()
            
            print(f"Created {len(saved_files)} visualizations")
            return saved_files
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description="Run CLAM regressor inference on WSI files")
    parser.add_argument("--wsi", required=True, help="Path to WSI file")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--model", default="checkpoints_regressor/cv_fold_0/best_model.pth", 
                       help="Path to trained model checkpoint (default: checkpoints_regressor/cv_fold_0/best_model.pth)")
    parser.add_argument("--force", action="store_true", 
                       help="Force regeneration of patches, features, and visualizations (ignore existing files)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Train a model first or specify a different model path")
        return
    
    print(f"Using model: {args.model}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wsi_path = Path(args.wsi)
    slide_name = wsi_path.stem
    
    features_dir = output_dir / "features"
    results_path = output_dir / f"{slide_name}_results.json"
    
    inference_engine = CLAMRegressorInference(
        model_path=args.model,
        config_path=None  # will use config from checkpoint
    )
    
    results = inference_engine.predict_from_wsi(
        wsi_path=str(args.wsi),
        features_dir=str(features_dir),
        extract_features=True,
        force=args.force
    )
    
    if results:
        import json
        
        if args.force and results_path.exists():
            print(f"Force regeneration - removing existing results: {results_path}")
            results_path.unlink()
        
        with open(results_path, 'w') as f:
            results_json = results.copy()
            results_json['attention_weights'] = results_json['attention_weights'].tolist()
            json.dump(results_json, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        if args.force:
            print(f"Force regeneration enabled - removing existing visualizations...")
            slide_name = Path(args.wsi).stem
            viz_patterns = [
                f"{slide_name}_attention_heatmap.png",
                f"{slide_name}_top_patches.png", 
                f"{slide_name}_attention_stats.png"
            ]
            
            for pattern in viz_patterns:
                viz_file = output_dir / pattern
                if viz_file.exists():
                    viz_file.unlink()
                    print(f"  Removed: {pattern}")
        
        viz_files = inference_engine.create_attention_visualization(
            str(args.wsi), results, output_dir
        )
        
        if viz_files:
            print(f"\nVisualization files created:")
            for viz_file in viz_files:
                print(f" {Path(viz_file).name}")
    else:
        print("Inference failed")


if __name__ == "__main__":
    main() 