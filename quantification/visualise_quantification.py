#!/usr/bin/env python

import argparse
import gc
from pathlib import Path
import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
from tiffslide import TiffSlide
import tifffile


def load_wsi_image(src: Path, size, downsample=8):
    if src.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
        img = cv2.imread(str(src))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(img)

    slide = TiffSlide(str(src))
    best_level = slide.get_best_level_for_downsample(downsample)
    region = slide.read_region((0, 0), best_level, slide.level_dimensions[best_level]).convert('RGB')
    img = np.array(region)

    if downsample > 1:
        h, w = img.shape[:2]
        new_size = (w // downsample, h // downsample)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(img)


def label_to_color(mask, sat=0.85, val=1.0):
    max_lab = int(mask.max())
    if max_lab == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)
    golden_ratio = 0.61803398875
    hues = (np.arange(1, max_lab + 1) * golden_ratio) % 1.0
    rgb = cm.hsv(hues)
    lut = (rgb[:, :3] * 255).astype(np.uint8)
    full_lut = np.zeros((max_lab + 1, 3), dtype=np.uint8)
    full_lut[1:max_lab + 1] = lut
    return full_lut[mask]


def blend_overlay(base, rgb_mask, alpha=0.4):
    overlay = Image.fromarray(rgb_mask).resize(base.size, Image.NEAREST)
    return Image.blend(base, overlay, alpha)


from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm

def visualize_inflammatory_cells(wsi_path, tubule_mask, cell_coords, output_dir, alpha=0.4, downsample=8):
    """
    Visualize inflammatory cells and tubules on the original WSI.
    
    Parameters:
        wsi_path (str): Path to the whole slide image.
        tubule_mask (np.ndarray): Array of tubule mask.
        cell_coords (np.ndarray): Array of cell coordinates.
        output_dir (str): Directory to save the output visualization.
        alpha (float): Transparency level for overlay.
        downsample (int): Downsampling factor.
    """
    from tiffslide import TiffSlide

    # Create output directory if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def label_to_color(mask, sat=0.85, val=1.0):
        max_lab = int(mask.max())
        if max_lab == 0:
            return np.zeros((*mask.shape, 3), dtype=np.uint8)
        golden_ratio = 0.61803398875
        hues = (np.arange(1, max_lab + 1) * golden_ratio) % 1.0
        rgb = cm.hsv(hues)
        lut = (rgb[:, :3] * 255).astype(np.uint8)
        full_lut = np.zeros((max_lab + 1, 3), dtype=np.uint8)
        full_lut[1:max_lab + 1] = lut
        return full_lut[mask]

    def load_wsi_image(wsi_path, size, downsample):
        slide = TiffSlide(wsi_path)
        best_level = slide.get_best_level_for_downsample(downsample)
        region = slide.read_region((0, 0), best_level, slide.level_dimensions[best_level]).convert('RGB')
        img = np.array(region)
        if downsample > 1:
            h, w = img.shape[:2]
            new_size = (w // downsample, h // downsample)
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(img)

    def blend_overlay(base, rgb_mask, alpha):
        overlay = Image.fromarray(rgb_mask).resize(base.size, Image.NEAREST)
        return Image.blend(base, overlay, alpha)
    
    # Load the WSI image
    wsi_image = load_wsi_image(wsi_path, tubule_mask.shape[:2], downsample)

    # Create colorful masks
    colorful_tubules = label_to_color(tubule_mask)

    # Create a point mask for inflammatory cells
    cell_mask = np.zeros_like(tubule_mask, dtype=np.uint8)
    for y, x in cell_coords:
        if 0 <= y < cell_mask.shape[0] and 0 <= x < cell_mask.shape[1]:
            cell_mask[y, x] = 255
    
    # Convert cell mask to RGB
    colorful_cells = cv2.applyColorMap(cell_mask, cv2.COLORMAP_JET)

    # Blend the tubule and cell overlays
    blended_tubules = blend_overlay(wsi_image, colorful_tubules, alpha)
    blended_cells = blend_overlay(blended_tubules, colorful_cells, alpha)

    # Save the final visualization
    visualization_path = output_dir / f"{Path(wsi_path).stem}_cell_tubule_visualization.png"
    blended_cells.save(visualization_path)
    print(f"Visualization saved to {visualization_path}")
    return str(visualization_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', required=True, help='Path to the WSI')
    parser.add_argument('--mask_dir', required=True, help='Directory containing instance masks')
    parser.add_argument('--alpha', type=float, default=0.4, help='Transparency of overlay')
    parser.add_argument('--downsample', type=int, default=8, help='Downsampling factor')
    args = parser.parse_args()

    visualize_inflammatory_cells(args.slide_path, args.mask_dir, alpha=args.alpha, downsample=args.downsample)
