# Credit to Claude LLM for patch extraction and feature extraction code

"""
UNI Foundation Model Feature Extractor
Extracts features from pathology patches using the UNI model
"""

import os
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from PIL import Image
import timm
from torchvision import transforms


class UNIFeatureExtractor:
    """Feature extractor using UNI foundation model for pathology"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['feature_extraction']['device'])
        self.batch_size = config['feature_extraction']['batch_size']
        self.precision = config['feature_extraction']['precision']
        
        # Initialize UNI model
        self.model = self._load_uni_model()
        self.transform = self._get_transform()
        
        print(f"UNI Feature Extractor initialized on {self.device}")
        
    def _load_uni_model(self) -> nn.Module:
        """Load UNI foundation model"""
        model_type = self.config['feature_extraction'].get('model', 'UNI')
        
        try:
            if model_type == 'UNI':
                # Try to load UNI from Hugging Face Hub (requires authentication and access approval)
                try:
                    print("Attempting to load UNI model from Hugging Face Hub...")
                    print("Note: UNI requires access approval at https://huggingface.co/MahmoodLab/UNI")
                    
                    # Load UNI from Hugging Face Hub with correct format
                    model = timm.create_model(
                        "hf_hub:MahmoodLab/UNI",  # correct format 
                        pretrained=True, 
                        init_values=1e-5, 
                        dynamic_img_size=True,
                        num_classes=0,  # Remove classification head
                        global_pool='token'  # Use [CLS] token
                    )
                    print("Successfully loaded UNI model from Hugging Face Hub")
                    
                except Exception as hf_error:
                    print(f"Could not load UNI model: {hf_error}")
                    if "403" in str(hf_error) or "unauthorized" in str(hf_error).lower():
                        print("ACCESS REQUIRED: Please visit https://huggingface.co/MahmoodLab/UNI to request access")
                        print("Once approved, the UNI model will work automatically")
                    print("Falling back to ImageNet ViT-Large (similar architecture)...")
                    raise hf_error
                    
            # elif model_type == 'CTransPath':
            #     # CTransPath - another pathology foundation model, may have better PAS generalization
            #     model = timm.create_model(
            #         'swin_tiny_patch4_window7_224',
            #         pretrained=False,
            #         num_classes=0
            #     )
                # Load CTransPath weights if available
                # checkpoint = torch.load('path/to/ctranspath.pth')
                # model.load_state_dict(checkpoint)
            else:  # default to ViT-Large
                print(f"Loading {model_type} model...")
                model = timm.create_model(
                    'vit_large_patch16_224',
                    pretrained=True,
                    num_classes=0,
                    global_pool='token'
                )
            
            model.eval()
            model.to(self.device)
            
            # Enable mixed precision if specified
            if self.precision == 'fp16':
                model = model.half()
            
            print(f"Loaded {model_type} model successfully")
            return model
            
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            print("Using ImageNet ViT-Large as fallback (1024-dim features, good for histopathology)")
            
            # Fallback to ImageNet pretrained ViT-Large
            model = timm.create_model(
                'vit_large_patch16_224',
                pretrained=True,
                num_classes=0,
                global_pool='token'
            )
            
            model.eval()
            model.to(self.device)
            
            if self.precision == 'fp16':
                model = model.half()
                
            print("ViT-Large fallback model loaded successfully")
            return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get preprocessing transforms for UNI model"""
        return transforms.Compose([
            transforms.Resize(224),  # UNI expects 224x224 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features_from_patches(self, patches: np.ndarray) -> np.ndarray:
        """Extract features from a batch of patches"""
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch_patches = patches[i:i + self.batch_size]
                
                # Preprocess patches
                batch_tensors = []
                for patch in batch_patches:
                    # Convert numpy array to PIL Image
                    patch_pil = Image.fromarray(patch.astype(np.uint8))
                    patch_tensor = self.transform(patch_pil)
                    batch_tensors.append(patch_tensor)
                
                # Stack into batch tensor
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                if self.precision == 'fp16':
                    batch_tensor = batch_tensor.half()
                
                # Extract features
                features = self.model(batch_tensor)
                # Keep as float32 for maximum model performance
                features = features.cpu().numpy().astype(np.float32)
                
                features_list.append(features)
        
        return np.vstack(features_list)
    
    def process_slide_patches(self, h5_path: str) -> Dict[str, Any]:
        """Process patches from a single slide HDF5 file"""
        slide_name = Path(h5_path).parent.name
        
        try:
            with h5py.File(h5_path, 'r') as f:
                patches = f['patches'][:]
                coordinates = f['coordinates'][:]
                metadata = {
                    'slide_name': f.attrs['slide_name'],
                    'num_patches': f.attrs['num_patches'],
                    'patch_size': f.attrs['patch_size'],
                    'level': f.attrs['level']
                }
            
            print(f"  Extracting features from {len(patches)} patches...")
            
            # Extract features
            features = self.extract_features_from_patches(patches)
            
            return {
                'slide_name': slide_name,
                'features': features,
                'coordinates': coordinates,
                'metadata': metadata,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"  Error processing {slide_name}: {e}")
            return {
                'slide_name': slide_name,
                'features': None,
                'coordinates': None,
                'metadata': None,
                'status': f'error: {e}'
            }
    
    def save_features(self, feature_data: Dict[str, Any], output_dir: Path) -> str:
        """Save extracted features to HDF5 file"""
        slide_name = feature_data['slide_name']
        features = feature_data['features']
        coordinates = feature_data['coordinates']
        metadata = feature_data['metadata']
        
        # Create slide-specific directory
        slide_dir = output_dir / slide_name
        slide_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features as HDF5 file
        h5_path = slide_dir / f"{slide_name}_features.h5"
        
        with h5py.File(h5_path, 'w') as f:
            if features is not None:
                f.create_dataset('features', data=features, compression='gzip')
                f.create_dataset('coordinates', data=coordinates)
                
                # Save metadata
                for key, value in metadata.items():
                    f.attrs[key] = value
                
                f.attrs['feature_dim'] = features.shape[1]
                f.attrs['model'] = 'UNI'
                f.attrs['status'] = feature_data['status']
        
        return str(h5_path)
    
    def process_all_slides(self, patch_dir: str, features_dir: str) -> Dict[str, Any]:
        """Process all slides and extract features"""
        patch_path = Path(patch_dir)
        features_path = Path(features_dir)
        features_path.mkdir(parents=True, exist_ok=True)
        
        # Find all patch HDF5 files
        h5_files = list(patch_path.glob("*/*_patches.h5"))
        
        if not h5_files:
            raise ValueError(f"No patch files found in {patch_dir}")
        
        results = {
            "processed_slides": [],
            "failed_slides": [],
            "total_features": 0,
            "summary": {}
        }
        
        print(f"Processing {len(h5_files)} slides for feature extraction...")
        
        for h5_path in tqdm(h5_files, desc="Extracting features"):
            feature_data = self.process_slide_patches(str(h5_path))
            
            if feature_data['status'] == 'success':
                features_h5_path = self.save_features(feature_data, features_path)
                results["processed_slides"].append({
                    "slide_name": feature_data['slide_name'],
                    "num_features": len(feature_data['features']),
                    "feature_dim": feature_data['features'].shape[1],
                    "h5_path": features_h5_path
                })
                results["total_features"] += len(feature_data['features'])
            else:
                results["failed_slides"].append({
                    "slide_name": feature_data['slide_name'],
                    "status": feature_data['status']
                })
        
        # Save summary
        results["summary"] = {
            "total_slides_processed": len(h5_files),
            "successful_slides": len(results["processed_slides"]),
            "failed_slides": len(results["failed_slides"]),
            "total_features_extracted": results["total_features"],
            "average_features_per_slide": results["total_features"] / max(1, len(results["processed_slides"])),
            "feature_dimension": results["processed_slides"][0]["feature_dim"] if results["processed_slides"] else 0
        }
        
        # Save results summary
        summary_path = features_path / "feature_extraction_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFeature extraction complete!")
        print(f"Successfully processed: {results['summary']['successful_slides']} slides")
        print(f"Total features extracted: {results['summary']['total_features_extracted']}")
        print(f"Feature dimension: {results['summary']['feature_dimension']}")
        
        return results


def download_uni_weights():
    """Download UNI model weights if not available"""
    try:
        import huggingface_hub
        
        # UNI model is available on Hugging Face
        model_name = "microsoft/uni"
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        
        print("Downloading UNI model weights...")
        huggingface_hub.snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            force_download=False
        )
        print("UNI weights downloaded successfully!")
        
    except ImportError:
        print("Please install huggingface_hub to download UNI weights:")
        print("pip install huggingface_hub")
    except Exception as e:
        print(f"Error downloading UNI weights: {e}")
        print("Please manually download from: https://huggingface.co/microsoft/uni")


if __name__ == "__main__":
    # Example usage
    config = {
        'feature_extraction': {
            'device': 'cuda',
            'batch_size': 32,
            'precision': 'fp16'
        }
    }
    
    extractor = UNIFeatureExtractor(config)
    results = extractor.process_all_slides(
        patch_dir="./data/patches",
        features_dir="./data/features"
    ) 