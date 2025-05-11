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

