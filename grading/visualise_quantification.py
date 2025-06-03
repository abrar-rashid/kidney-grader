"""
quantification/visualize_quantification.py
------------------------------------------

Create a single overlay that shows

    ▸ the original WSI (RGB)
    ▸ the colourised *tubule* instance mask (semi-transparent)
    ▸ inflammatory-cell centroids (small yellow circles)

Memory-efficient – it never loads the full-resolution WSI into RAM/VRAM;
instead it chooses the finest built-in pyramid level whose largest
dimension does not exceed `max_dim` (default ≈ 8 k px).

"""
from __future__ import annotations
import math
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import tifffile
from tiffslide import TiffSlide

# Add the parent directory to sys.path to allow relative imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from segmentation.segment import colourise_instances, overlay_rgb

# Constant that appears elsewhere in the code-base
MICRONS_PER_PIXEL = 0.24199951445730394


def _auto_level(slide: TiffSlide, max_dim: int) -> Tuple[int, float]:
    """
    Pick the slide pyramid level whose dimensions are *just* below max_dim.

    Returns
    -------
    level : int
        Pyramid level index (0 = full-res).
    downsample : float
        Total down-sampling factor relative to level 0.
    """
    widths, heights = zip(*slide.level_dimensions)
    for lvl, (w, h) in enumerate(slide.level_dimensions):
        if max(w, h) <= max_dim:
            return lvl, slide.level_downsamples[lvl]
    # Fall back to the coarsest level
    lvl = slide.level_count - 1
    return lvl, slide.level_downsamples[lvl]


def _coords_mm_to_pixel(coords_mm: np.ndarray) -> np.ndarray:
    """Convert N × 2 array of (x_mm, y_mm) → (x_px, y_px) at level 0."""
    return (coords_mm * 1000.0 / MICRONS_PER_PIXEL).astype(np.int64)


def _ensure_same_size(ref: Image.Image, img: Image.Image) -> Image.Image:
    """Return *img* resized / padded so that img.size == ref.size."""
    if ref.size == img.size:
        return img

    rw, rh = ref.size
    iw, ih = img.size

    # ── Case A: difference ≤ -2 px → just resize (nearest keeps instance ids intact)
    if abs(rw - iw) <= 2 and abs(rh - ih) <= 2:
        return img.resize((rw, rh), Image.NEAREST)

    # ── Case B: larger mismatch → crop or pad top-left aligned
    #            (masks produced by the pipeline always start at (0,0)).
    new = Image.new("RGB", ref.size)
    crop = img.crop((0, 0, min(rw, iw), min(rh, ih)))
    new.paste(crop, (0, 0))
    return new


def create_quantification_overlay(
    *,
    wsi_path: str | Path,
    tubule_mask: np.ndarray | str | Path,
    cell_coords: np.ndarray,
    output_dir: str | Path,
    cell_coords_in_pixels: bool = False,
    max_dim: int = 50000,
    tubule_alpha: float = 0.35,
    point_radius_px: int = 4,
) -> Optional[Path]:

    wsi_path = Path(wsi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. read WSI thumbnail
    slide = TiffSlide(str(wsi_path))
    lvl, ds = _auto_level(slide, max_dim)
    downsample = int(round(ds))
    w_lvl, h_lvl = slide.level_dimensions[lvl]
    wsi_rgb = slide.read_region((0, 0), lvl, (w_lvl, h_lvl)).convert("RGB")

    # ── 2. read tubule mask & down-sample == stride (no interpolation)
    if isinstance(tubule_mask, (str, Path)):
        with tifffile.TiffFile(str(tubule_mask)) as tif:
            tubule_mask = tif.asarray(out="memmap")

    tubule_mask_ds = tubule_mask[::downsample, ::downsample] if downsample > 1 else tubule_mask
    mask_rgb = colourise_instances(tubule_mask_ds, shuffle=True)

    # **NEW** → guarantee identical size before blending
    mask_rgb = _ensure_same_size(wsi_rgb, mask_rgb)

    overlay_tub = overlay_rgb(wsi_rgb, mask_rgb, alpha=tubule_alpha)

    # ── 3. draw inflammatory-cell centroids
    cell_coords_px = (
        _coords_mm_to_pixel(cell_coords) if not cell_coords_in_pixels
        else cell_coords.astype(np.int64)
    )
    pts_ds = (cell_coords_px / downsample).round().astype(np.int64)

    draw = ImageDraw.Draw(overlay_tub)
    for x, y in pts_ds:
        if 0 <= x < w_lvl and 0 <= y < h_lvl:
            draw.ellipse(
                (x - point_radius_px, y - point_radius_px,
                 x + point_radius_px, y + point_radius_px),
                outline=(255, 255, 0),
                width=max(1, point_radius_px // 2),
            )

    # ── 4. save
    out_png  = output_dir / f"{wsi_path.stem}_tubule_inflam_overlay_lvl{lvl}.png"
    out_tiff = out_png.with_suffix(".tiff")

    overlay_tub.save(out_png, optimize=True)

    # ----- NEW: also write a TIFF for QuPath ---------------------------
    import tifffile
    tifffile.imwrite(
        out_tiff,
        np.asarray(overlay_tub),          # RGB uint8
        photometric="rgb",
        bigtiff=True,                     # >4 GB safety; harmless if small
        compression=("zstd", 5),          # fast & widely supported
    )
    # -------------------------------------------------------------------

    return out_png


# ---------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse, numpy as np, json
    p = argparse.ArgumentParser(description="Overlay tubules + inflammatory cells on WSI")
    p.add_argument("--wsi", required=True, help=".svs/.tif file")
    p.add_argument("--tubule_mask", required=True, help="Path to class-1 instance mask (.tiff)")
    p.add_argument("--inflam_json", required=True,
                   help="JSON with inflammatory-cell detection (as produced in Stage 2)")
    p.add_argument("--output_dir", default=".", help="Where to write the PNG")
    p.add_argument("--prob_thres", type=float, default=0.50,
                   help="Keep only detections with probability ≥ this")
    p.add_argument("--max_dim", type=int, default=50000, help="Longest side of overlay")
    args = p.parse_args()

    with open(args.inflam_json) as f:
        data = json.load(f)
    pts = np.array([[p["point"][0], p["point"][1]]
                    for p in data["points"]
                    if "probability" not in p or p["probability"] >= args.prob_thres],
                   dtype=np.float32)

    out = create_quantification_overlay(
        wsi_path=args.wsi,
        tubule_mask=args.tubule_mask,
        cell_coords=pts,
        cell_coords_in_pixels=False,   # JSON stores mm
        output_dir=args.output_dir,
        max_dim=args.max_dim
    )
    print(f"Overlay written → {out}")
