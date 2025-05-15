#!/usr/bin/env python
"""
Visualize instance masks as colorful overlays.

For each class mask saved by the KidneyGrader pipeline (e.g., *_full_instance_mask_class1.npy), this script:
1. Counts the labeled objects.
2. Creates an RGB mask where each object has a unique color.
3. Blends the mask with the original image.
4. Saves the visualization as <slide_stem>_class{N}_instances.png.

It can also create a merged overlay for multiple classes using --write_merged.
"""

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
from tiffslide import TiffSlide
import tifffile

# ---------------------------------------------------------------------------
def load_source_image(src: Path, size, downsample=8):
    """Load the WSI at the appropriate resolution and downsample."""
    if src.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        img = cv2.imread(str(src))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(img)

    slide = TiffSlide(str(src))
    best_level = slide.get_best_level_for_downsample(downsample)
    level_w, level_h = slide.level_dimensions[best_level]

    # Load directly at the best level for efficiency
    region = slide.read_region((0, 0), best_level, (level_w, level_h)).convert("RGB")
    img = np.array(region)

    # Downsample using the same method as the overlay
    if downsample > 1:
        h, w = img.shape[:2]
        new_size = (w // downsample, h // downsample)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(img)

def downsample_array(array, factor):
    """Downsample a numpy array."""
    return cv2.resize(array, (array.shape[1] // factor, array.shape[0] // factor), interpolation=cv2.INTER_NEAREST)

def label_to_color(mask: np.ndarray, sat=0.85, val=1.0):
    """
    Return an RGB uint8 array the same shape as `mask`, coloring each
    instance with a unique hue.
    """
    max_lab = int(mask.max())
    if max_lab == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    # Build a color look-up table
    golden = 0.61803398875
    labels = np.arange(1, max_lab + 1, dtype=np.float32)
    hues = (labels * golden) % 1.0
    sats = np.full_like(hues, sat)
    vals = np.full_like(hues, val)

    # Use HSV to RGB conversion directly
    hsv = np.stack([hues, sats, vals], axis=1)  # (L, 3)
    rgb_float = mcolors.hsv_to_rgb(hsv)  # Convert to RGB
    lut = (rgb_float * 255).astype(np.uint8)  # (L, 3)

    # Create a full lookup table with zero as background color
    full_lut = np.zeros((max_lab + 1, 3), dtype=np.uint8)
    full_lut[1:max_lab + 1] = lut

    # Map every pixel label to color
    rgb_img = full_lut[mask]

    return rgb_img


def blend(base: Image.Image, rgb_mask: np.ndarray, alpha=0.4):
    """Blend the mask onto the base image."""
    overlay = Image.fromarray(rgb_mask).resize(base.size, Image.NEAREST)
    return Image.blend(base, overlay, alpha)

# ---------------------------------------------------------------------------
def main(slide_path: str, case_dir: str, classes=(1, 4), alpha=0.4, downsample=8, write_merged=False):

    slide_path = Path(slide_path)
    case_dir = Path(case_dir)

    masks = {}
    for cls in classes:

        mask_path = case_dir / f"{slide_path.stem}_full_instance_mask_class{cls}.tiff"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask for class {cls} not found at {mask_path}")

        # Load the instance mask from BigTIFF
        with tifffile.TiffFile(mask_path) as tif:
            masks[cls] = tif.asarray(out='memmap')  # Use memory-mapped loading for efficiency

    for cls, mask in masks.items():
        n_instances = int(mask.max())
        print(f"Class {cls}: {n_instances} instances")

        # Generate a colorful mask
        colour_mask = label_to_color(mask)

        # Downsample for visualization
        small_mask = downsample_array(colour_mask, downsample)

        # Load the WSI at the appropriate size
        src = load_source_image(slide_path, size=(small_mask.shape[1], small_mask.shape[0]), downsample=downsample)

        # Blend and save the output
        out_img = blend(src, small_mask, alpha=alpha)
        out_path = case_dir / f"{slide_path.stem}_class{cls}_instances.png"
        out_img.save(out_path)
        print(f"Saved visualization at {out_path}")

        del colour_mask, out_img, src
        gc.collect()

    # Merged overlay for multiple classes
    if write_merged:
        merged = np.zeros_like(next(iter(masks.values())), dtype=np.uint8)
        for cls, m in masks.items():
            merged[m > 0] = cls
        
        small_merged = downsample_array(merged, downsample)
        src = load_source_image(slide_path, size=(small_merged.shape[1], small_merged.shape[0]), downsample=downsample)

        cmap = cm.get_cmap("nipy_spectral", int(max(classes)) + 1)
        colour = (cmap(small_merged / small_merged.max())[..., :3] * 255).astype(np.uint8)
        out_img = blend(src, colour, alpha=alpha)
        out_path = case_dir / f"{slide_path.stem}_merged_overlay.png"
        out_img.save(out_path)
        print(f"Saved merged overlay at {out_path}")

    print("Processing complete.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_path", required=True, help="Original WSI or image file")
    ap.add_argument("--case_dir", required=True, help="Folder holding instance masks")
    ap.add_argument("--classes", default="1,4", help="Comma-separated class indices")
    ap.add_argument("--alpha", type=float, default=0.4, help="Overlay transparency")
    ap.add_argument("--downsample", type=int, default=8, help="Downsample factor")
    ap.add_argument("--write_merged", action="store_true", help="Create merged overlay")
    args = ap.parse_args()

    class_tuple = tuple(int(c) for c in args.classes.split(",") if c.strip())
    main(args.slide_path, args.case_dir, classes=class_tuple, alpha=args.alpha, downsample=args.downsample, write_merged=args.write_merged)
