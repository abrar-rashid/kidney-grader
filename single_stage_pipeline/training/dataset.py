# Credit to Claude LLM for the dataset handling code

"""
Dataset class for loading WSI features and labels for CLAM training
"""

import os
import json
import h5py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from torch.utils.data import Dataset


class WSIFeaturesDataset(Dataset):
    """Dataset for loading WSI features and tubulitis labels"""
    
    def __init__(
        self,
        features_dir: str,
        labels_file: str,
        split_file: str,
        split_name: str,
        target_column: str = "T",
        max_patches: Optional[int] = None,
        augment: bool = False
    ):
        """
        Args:
            features_dir: Directory containing feature HDF5 files
            labels_file: CSV file with labels
            split_file: JSON file with train/val/test splits
            split_name: Which split to load ('train', 'val', 'test')
            target_column: Column name for target scores
            max_patches: Maximum number of patches per WSI (for memory efficiency)
            augment: Whether to apply feature augmentation (for training)
        """
        self.features_dir = Path(features_dir)
        self.target_column = target_column
        self.max_patches = max_patches
        self.augment = augment
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        
        # Load split information
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        self.slide_names = splits[split_name]
        print(f"Loaded {len(self.slide_names)} slides for {split_name} split")
        
        # Filter labels for this split
        self.split_labels = self.labels_df[
            self.labels_df['filename'].isin(self.slide_names)
        ].copy()
        
        # Remove slides without target labels
        self.split_labels = self.split_labels.dropna(subset=[target_column])
        self.slide_names = self.split_labels['filename'].tolist()
        
        print(f"Final dataset size: {len(self.slide_names)} slides with valid labels")
        
        # Validate that feature files exist
        self._validate_feature_files()
        
    def _validate_feature_files(self):
        """Validate that feature files exist for all slides"""
        valid_slides = []
        missing_features = []
        
        for slide_name in self.slide_names:
            slide_name_no_ext = Path(slide_name).stem
            feature_file = self.features_dir / slide_name_no_ext / f"{slide_name_no_ext}_features.h5"
            
            if feature_file.exists():
                valid_slides.append(slide_name)
            else:
                missing_features.append(slide_name)
        
        if missing_features:
            print(f"Warning: Missing feature files for {len(missing_features)} slides")
            print(f"First few missing: {missing_features[:3]}")
        
        self.slide_names = valid_slides
        self.split_labels = self.split_labels[
            self.split_labels['filename'].isin(valid_slides)
        ]
        
        print(f"Dataset after validation: {len(self.slide_names)} slides")
    
    def __len__(self) -> int:
        return len(self.slide_names)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single WSI's features and label"""
        slide_name = self.slide_names[idx]
        slide_name_no_ext = Path(slide_name).stem
        
        # Load features
        feature_file = self.features_dir / slide_name_no_ext / f"{slide_name_no_ext}_features.h5"
        
        try:
            with h5py.File(feature_file, 'r') as f:
                features = f['features'][:]  # [num_patches, feature_dim]
                coordinates = f['coordinates'][:]  # [num_patches, 2]
                
                # Get metadata
                metadata = {
                    'slide_name': f.attrs.get('slide_name', slide_name_no_ext),
                    'num_patches': f.attrs.get('num_patches', len(features)),
                    'feature_dim': f.attrs.get('feature_dim', features.shape[1])
                }
        
        except Exception as e:
            print(f"Error loading features for {slide_name}: {e}")
            # Return dummy data
            features = np.zeros((1, 1024), dtype=np.float32)
            coordinates = np.zeros((1, 2), dtype=np.int32)
            metadata = {'slide_name': slide_name_no_ext, 'num_patches': 1, 'feature_dim': 1024}
        
        # Get label
        label_row = self.split_labels[self.split_labels['filename'] == slide_name]
        if len(label_row) > 0:
            label = float(label_row[self.target_column].iloc[0])
        else:
            print(f"Warning: No label found for {slide_name}")
            label = 0.0
        
        # Apply max patches limit
        if self.max_patches and len(features) > self.max_patches:
            # Randomly sample patches during training, take first N during inference
            if self.augment:
                indices = np.random.choice(len(features), self.max_patches, replace=False)
            else:
                indices = np.arange(self.max_patches)
            
            features = features[indices]
            coordinates = coordinates[indices]
            metadata['num_patches'] = self.max_patches
        
        # Apply feature augmentation for training
        if self.augment:
            features = self._augment_features(features)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        coordinates_tensor = torch.from_numpy(coordinates).long()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return {
            'features': features_tensor,
            'coordinates': coordinates_tensor,
            'label': label_tensor,
            'slide_name': slide_name_no_ext,
            'metadata': metadata
        }
    
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """Enhanced feature augmentation for small datasets"""
        augmented_features = features.copy()
        
        # 1. Add Gaussian noise (simulates slight variations in tissue/staining)
        if np.random.random() < 0.5:
            noise_scale = 0.02  # Small noise to prevent mode collapse
            noise = np.random.normal(0, noise_scale, features.shape)
            augmented_features = augmented_features + noise.astype(features.dtype)
        
        # 2. Feature dropout (simulates missing/corrupted patches)
        if np.random.random() < 0.3:
            dropout_rate = 0.1
            mask = np.random.random(features.shape) > dropout_rate
            augmented_features = augmented_features * mask
        
        # 3. Feature scaling variation (simulates staining intensity differences)
        if np.random.random() < 0.2:
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented_features = augmented_features * scale_factor
        
        # 4. Patch reordering (simulates different spatial sampling)
        if np.random.random() < 0.3:
            num_patches = features.shape[0]
            if num_patches > 1:
                # Randomly shuffle a subset of patches
                shuffle_ratio = 0.3
                num_to_shuffle = int(num_patches * shuffle_ratio)
                shuffle_indices = np.random.choice(num_patches, num_to_shuffle, replace=False)
                shuffled_order = np.random.permutation(shuffle_indices)
                augmented_features[shuffle_indices] = augmented_features[shuffled_order]
        
        return augmented_features
    
    def get_label_distribution(self) -> Dict[float, int]:
        """Get distribution of labels in this split"""
        labels = self.split_labels[self.target_column].values
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels, counts))


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for WSI features
    Since each WSI has different number of patches, we need special handling
    """
    # For MIL, batch size is typically 1, but we handle variable batch sizes
    if len(batch) == 1:
        # Single sample - just add batch dimension
        sample = batch[0]
        return {
            'features': sample['features'].unsqueeze(0),  # [1, num_patches, feature_dim]
            'coordinates': sample['coordinates'].unsqueeze(0),  # [1, num_patches, 2]
            'labels': sample['label'].unsqueeze(0),  # [1]
            'slide_names': [sample['slide_name']],
            'metadata': [sample['metadata']]
        }
    
    else:
        # Multiple samples - need to pad or handle variable lengths
        features_list = []
        coordinates_list = []
        labels_list = []
        slide_names = []
        metadata_list = []
        
        max_patches = max(sample['features'].shape[0] for sample in batch)
        feature_dim = batch[0]['features'].shape[1]
        
        for sample in batch:
            features = sample['features']
            coordinates = sample['coordinates']
            num_patches = features.shape[0]
            
            # Pad with zeros if necessary
            if num_patches < max_patches:
                pad_size = max_patches - num_patches
                features_pad = torch.zeros(pad_size, feature_dim)
                coords_pad = torch.zeros(pad_size, 2, dtype=torch.long)
                
                features = torch.cat([features, features_pad], dim=0)
                coordinates = torch.cat([coordinates, coords_pad], dim=0)
            
            features_list.append(features)
            coordinates_list.append(coordinates)
            labels_list.append(sample['label'])
            slide_names.append(sample['slide_name'])
            metadata_list.append(sample['metadata'])
        
        return {
            'features': torch.stack(features_list),  # [batch_size, max_patches, feature_dim]
            'coordinates': torch.stack(coordinates_list),  # [batch_size, max_patches, 2]
            'labels': torch.stack(labels_list),  # [batch_size]
            'slide_names': slide_names,
            'metadata': metadata_list
        }


def create_data_loaders(
    features_dir: str,
    labels_file: str,
    split_file: str,
    config: Dict[str, Any]
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test data loaders"""
    
    data_config = config['data']
    
    # Create datasets
    train_dataset = WSIFeaturesDataset(
        features_dir=features_dir,
        labels_file=labels_file,
        split_file=split_file,
        split_name='train',
        target_column=data_config['target_column'],
        max_patches=None,  # Use all patches for training
        augment=True
    )
    
    val_dataset = WSIFeaturesDataset(
        features_dir=features_dir,
        labels_file=labels_file,
        split_file=split_file,
        split_name='val',
        target_column=data_config['target_column'],
        max_patches=None,
        augment=False
    )
    
    test_dataset = WSIFeaturesDataset(
        features_dir=features_dir,
        labels_file=labels_file,
        split_file=split_file,
        split_name='test',
        target_column=data_config['target_column'],
        max_patches=None,
        augment=False
    )
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Train: {len(train_dataset)} slides")
    print(f"Val: {len(val_dataset)} slides")
    print(f"Test: {len(test_dataset)} slides")
    
    print(f"\nTrain label distribution: {train_dataset.get_label_distribution()}")
    print(f"Val label distribution: {val_dataset.get_label_distribution()}")
    print(f"Test label distribution: {test_dataset.get_label_distribution()}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 