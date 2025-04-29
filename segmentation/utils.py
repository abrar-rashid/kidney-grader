import os
import time
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from tiffslide import TiffSlide
from torchvision import transforms
import torch
import logging
from scipy import ndimage
from skimage.morphology import local_maxima
from skimage.segmentation import watershed
from .improved_unet import ImprovedUNet
from .config import LABEL_COLOURS, PATCH_SIZE, PATCH_OVERLAP, TISSUE_THRESHOLD, PATCH_LEVEL, NUM_CLASSES

import numpy as np
import cv2
import logging

import numpy as np
import cv2
import logging
from skimage.measure import label

def get_instance_mask(mask: np.ndarray, tissue_type: int = 1, downsample_factor: int = 4) -> tuple[np.ndarray, int]:
    binary = (mask == tissue_type).astype(np.uint8)

    if np.count_nonzero(binary) == 0:
        return np.zeros_like(binary, dtype=np.uint16), 0

    # morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if np.count_nonzero(opened) == 0:
        logging.warning("[Watershed-DS] Opened mask is empty.")
        return np.zeros_like(opened, dtype=np.uint16), 0

    # downsample for processing
    if downsample_factor > 1:
        opened_small = cv2.resize(opened, (
            opened.shape[1] // downsample_factor,
            opened.shape[0] // downsample_factor
        ), interpolation=cv2.INTER_NEAREST)
    else:
        opened_small = opened

    # distance transform
    dist = cv2.distanceTransform(opened_small, cv2.DIST_L2, 5).astype(np.float32)

    _, sure_fg = cv2.threshold(dist, 0.25 * dist.max(), 1, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opened_small, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    color_img = cv2.cvtColor((opened_small * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    labels = np.where(markers > 1, markers - 1, 0).astype(np.uint16)

    # upsample to original size
    if downsample_factor > 1:
        instance_mask = cv2.resize(
            labels,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        instance_mask = labels

    num_instances = instance_mask.max()
    logging.info(f"Found {num_instances} instances for tissue type {tissue_type}")
    return instance_mask, num_instances


def get_binary_class_mask(mask, class_idx):
    # convert multi-class mask to binary mask for a specific class.
    return (mask == class_idx).astype(np.uint8)


def create_visualization(mask, original_image=None, alpha=0.4):
    #Create color-coded mask and optionally overlay it on the original image
    h, w = mask.shape
    colour_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, colour in LABEL_COLOURS.items():
        colour_mask[mask == label_id] = colour
    colour_mask = Image.fromarray(colour_mask)
    
    if original_image is not None:
        original = original_image.convert("RGBA")
        mask_rgba = colour_mask.convert("RGBA")
        return Image.blend(original, mask_rgba, alpha)
    return colour_mask


def load_model(checkpoint_path, device, weights_only=True, num_classes=NUM_CLASSES):
    model = ImprovedUNet(n_classes=num_classes).to(device)
    if weights_only:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    return model


class PatchFolderDataset(Dataset):
    #dataset class for loading patches from a folder
    def __init__(self, patch_dir, transform=None):
        self.paths = [os.path.join(patch_dir, f"patch_{i}.png") for i in range(len(os.listdir(patch_dir))) 
              if os.path.exists(os.path.join(patch_dir, f"patch_{i}.png"))]

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8, 0.65, 0.8], std=[0.15, 0.15, 0.15])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def extract_patches_from_wsi(wsi_path, out_dir, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP,
                             level=PATCH_LEVEL, tissue_threshold=TISSUE_THRESHOLD, 
                             tissue_filtering=False, low_res_level=2, save_patches=True):
    # Extracts patches from a Whole Slide Image and save them to disk
    os.makedirs(out_dir, exist_ok=True)
    slide = TiffSlide(wsi_path)
    width, height = slide.level_dimensions[level]

    stride = int(round(patch_size * (1 - overlap)))
    slide_map = {}
    inference_flags = {}
    count = 0  # Total number of patches
    patch_index = 0  # Number of saved (foreground) patches

    from tqdm import tqdm

    # generate tissue mask at low resolution
    if tissue_filtering:
        logging.info("Generating low-resolution tissue mask for filtering...")
        low_res_img = slide.read_region((0, 0), level=low_res_level, size=slide.level_dimensions[low_res_level]).convert("RGB")
        low_res_np = np.array(low_res_img)
        tissue_mask_low = is_tissue_patch(low_res_np, threshold=tissue_threshold, return_mask=True)

        # upsampling to match the desired level
        tissue_mask = cv2.resize(
            tissue_mask_low.astype(np.uint8),
            slide.level_dimensions[level],
            interpolation=cv2.INTER_NEAREST
        )
    else:
        tissue_mask = None

    coords = [
        (x, y)
        for y in range(0, height - patch_size + 1, stride)
        for x in range(0, width - patch_size + 1, stride)
    ]

    for x, y in tqdm(coords, desc="Extracting patches", unit="patch"):
        # skip using tissue mask before reading patch
        if tissue_filtering:
            patch_mask = tissue_mask[y:y+patch_size, x:x+patch_size]
            tissue_fraction = np.mean(patch_mask)
            should_infer = tissue_fraction > 0.05  # adjust threshold if needed
        else:
            should_infer = True

        if should_infer:
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")

            if save_patches:
                patch_path = os.path.join(out_dir, f"patch_{patch_index}.png")
                patch.save(patch_path, icc_profile=None)

            patch_index += 1

        # track original slide coordinate even for skipped patches
        slide_map[count] = (x, y, patch_size, patch_size)
        inference_flags[count] = should_infer
        count += 1

    # save tissue mask overview
    patch_mask = np.zeros((height, width), dtype=np.uint8)
    for i, (x, y, w, h) in slide_map.items():
        if inference_flags.get(i, False):
            patch_mask[y:y+h, x:x+w] = 1

    Image.fromarray((patch_mask * 255).astype(np.uint8)).save(os.path.join(out_dir, "patch_inference_mask.png"))

    logging.info(f"Total patches: {count}, Foreground patches processed: {patch_index}")
    return slide_map, inference_flags


def is_tissue_patch(patch_np, threshold=0.05, return_mask=False):
    # TODO FIX
    # determine if a patch contains enough tissue using HSV color space analysis
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    tissue_mask = (saturation > 20) & (value < 230)
    if return_mask:
        return tissue_mask.astype(np.uint8)
    tissue_percentage = np.sum(tissue_mask) / (patch_np.shape[0] * patch_np.shape[1])
    return tissue_percentage > threshold

def load_all_patches_in_folder(patch_dir, transform=None):
    return PatchFolderDataset(patch_dir, transform)
