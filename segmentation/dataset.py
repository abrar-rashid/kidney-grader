import h5py
from detection.patch_extractor import is_tissue_patch
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_augmentations():
    """
    Cleaned-up augmentations suitable for kidney biopsy segmentation tasks.
    """
    return A.Compose([
        # spatial augmentations
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ], p=0.5),
        A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-30, 30), p=0.5),
        A.ElasticTransform(alpha=120, sigma=6.0, p=0.5),

        # color augmentations specific to H&E stained tissue
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10),
        ], p=0.5),

        # simulate staining variations often seen in histology
        A.OneOf([
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.ChannelShuffle(p=0.1),
        ], p=0.5),

        # simulate artifacts common in WSI scanning
        A.OneOf([
            A.GaussNoise(mean=0, var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.4),

        # enhance certain tissue structures
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # final preprocessing
        A.Normalize(
            mean=[0.8, 0.65, 0.8],
            std=[0.15, 0.15, 0.15],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_validation_augmentations():
    return A.Compose([
        A.Normalize(
            mean=[0.8, 0.65, 0.8],
            std=[0.15, 0.15, 0.15],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

class H5SegmentationDataset(Dataset):
    def __init__(self, h5_path, transform=None, is_training=False, tissue_threshold=0.05):
        self.h5 = h5py.File(h5_path, 'r')
        self.images = self.h5['data']
        self.labels = self.h5['labels']
        self.transform = transform
        
        if self.transform is None:
            if is_training:
                self.transform = get_training_augmentations()
            else:
                self.transform = get_validation_augmentations()
        
        # filter patches with sufficient tissue content
        self.valid_indices = self._filter_patches(tissue_threshold) if tissue_threshold > 0 else list(range(len(self.images)))
    
    def _filter_patches(self, threshold):
        valid_indices = []
        for i in range(len(self.images)):
            img = self.images[i]
            # simple tissue detection assuming non-white pixels are tissue
            if is_tissue_patch(img, threshold):
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            orig_idx = self.valid_indices[idx]
            img = self.images[orig_idx].astype(np.uint8)  # (H, W, 3)
            mask = self.labels[orig_idx].astype(np.int64)  # (H, W)

            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask'].long()
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                if not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask)
            else:
                img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0
                mask = torch.from_numpy(mask)

            return img, mask
        
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            return None  # fallback to let collate_fn filter out erroneous indices

def safe_collate(batch): # custom collate function for use by dataloaders
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.default_collate(batch)
