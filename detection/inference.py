#!/usr/bin/env python3
"""Improved inflammatory‑cell inference script
------------------------------------------------
Fixes the remaining runtime errors that were triggered by
1. PyTorch slice stepping ("step must be greater than zero")
2. Sparse‑matrix centroid calculation producing 3‑D tensors.

The script now
- avoids negative‑step slicing on torch tensors by swapping the last two
  columns explicitly (`ctrs[:, (1,0)]`).
- implements a *safe_centroids* helper that does **not** rely on the buggy
  `torch.sparse.mm` call inside InstanSeg.
- maintains the earlier fixes that guaranteed 2‑D label maps and avoided
  the sparse‑mm path entirely.

"""
# ────────────────────────────────────────────────────────────────────
import os, sys, json, argparse
from pathlib import Path
import numpy as np

import torch
import ttach as tta
from tiffslide import TiffSlide

# make InstanSeg package importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from instanseg import InstanSeg
from instanseg.inference_class import _rescale_to_pixel_size
from instanseg.utils.pytorch_utils import (
    torch_fastremap,                     # safe utility
    _to_tensor_float32,                  # safe utility
)

# ── CONSTANTS ──────────────────────────────────────────────────────
INSTANSEG_MODEL        = "instanseg_brightfield_monkey.pt"
MODEL_NAMES            = [
    "1952372.pt",
    "1950672.pt",
    "1949389_2.pt",
]
DESTINATION_PIXEL_SIZE = 0.5
PATCH_SIZE             = 128
USE_TTA                = True
ORIGINAL_PIXEL_SIZE    = 0.24199951445730394
RESCALE_OUTPUT         = DESTINATION_PIXEL_SIZE != 0.5

# ────────────────────────────────────────────────────────────────────
# Helper: centroid computation *without* sparse.mm
@torch.no_grad()
def safe_centroids(label: torch.Tensor):
    """Return centroids (N,2) *y,x* and their label IDs (N,) for a 2‑D label map."""
    label = label.long()
    ids   = torch.unique(label)
    ids   = ids[ids != 0]  # skip background
    if ids.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=label.device), ids

    centroids = []
    for lid in ids:
        ys, xs = torch.where(label == lid)
        centroids.append(torch.tensor([ys.float().mean(), xs.float().mean()],
                                      device=label.device))
    return torch.stack(centroids), ids

# ────────────────────────────────────────────────────────────────────
class ModelEnsemble(torch.nn.Module):
    """Ensemble of multiple classification models with optional test-time augmentation."""
    def __init__(self, paths, device="cuda", use_tta=False):
        super().__init__()
        self.models = torch.nn.ModuleList([
            self._load(p, device, use_tta) for p in paths
        ])

    @staticmethod
    def _load(path, device, use_tta):
        m = torch.jit.load(path).eval().to(device)
        if use_tta:
            m = tta.ClassificationTTAWrapper(
                m,
                tta.Compose([
                    tta.VerticalFlip(),
                    tta.Rotate90([0, 90, 180, 270]),
                ]),
                merge_mode="mean",
            )
        return m

    def forward(self, x):
        with torch.no_grad():
            outs = [m(x) for m in self.models]
        return torch.mean(torch.stack(outs, 0), 0)

# ────────────────────────────────────────────────────────────────────
# CLI

def parse_cli():
    p = argparse.ArgumentParser("Inflammatory‑cell inference")
    p.add_argument("--wsi_path",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--bboxes", type=float, nargs="+", help="x1 y1 x2 y2 ...")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────
# Model loading

def load_models(model_dir):
    instanseg = InstanSeg(
        torch.jit.load(Path(model_dir)/INSTANSEG_MODEL).to("cuda"), verbosity=0
    )
    ensemble  = ModelEnsemble(
        [Path(model_dir)/n for n in MODEL_NAMES],
        device="cuda", use_tta=USE_TTA
    )
    return instanseg, ensemble

# ────────────────────────────────────────────────────────────────────
# Patch extraction without sparse.mm path

def masked_patches(label: torch.Tensor, rgb: torch.Tensor, patch_size: int = PATCH_SIZE):
    """Return masked crops & binary masks for *every* instance label."""
    # ensure 2‑D
    label = label.squeeze()
    if label.ndim > 2:
        label = label.view(-1, *label.shape[-2:])[0]
    assert label.ndim == 2

    cent, ids = safe_centroids(label)
    if ids.numel() == 0:
        empty = torch.empty((0, 3, patch_size, patch_size))
        return empty, empty[:, :1]

    pad = patch_size // 2
    rgb_p   = torch.nn.functional.pad(rgb,  (pad, pad, pad, pad))
    lab_p   = torch.nn.functional.pad(label, (pad, pad, pad, pad))

    crops, masks = [], []
    for (y, x), lid in zip(cent.long(), ids):
        y += pad; x += pad
        crops.append(rgb_p[:,  y-pad:y+pad,   x-pad:x+pad])
        masks.append((lab_p[y-pad:y+pad, x-pad:x+pad] == lid)
                      .unsqueeze(0).float())
    return torch.stack(crops), torch.stack(masks)

# ────────────────────────────────────────────────────────────────────
# Single‑bbox processing

def process_bbox(slide, bbox, instanseg, classifier):
    x1, y1, x2, y2 = bbox
    w,  h          = x2 - x1, y2 - y1
    region         = slide.read_region((y1, x1), 0, (h, w), as_array=True)

    # ── InstanSeg segmentation
    labels, _ = instanseg.eval_medium_image(
        region, pixel_size=ORIGINAL_PIXEL_SIZE,
        rescale_output=RESCALE_OUTPUT, seed_threshold=0.1, tile_size=1024
    )
    labels = torch_fastremap(labels.cpu()).squeeze()
    if labels.ndim == 3:
        labels = labels[0] if labels.shape[0] <= 3 else labels[..., 0]

    # ── RGB tensor rescaled to DESTINATION_PIXEL_SIZE
    rgb = _rescale_to_pixel_size(
        _to_tensor_float32(region), ORIGINAL_PIXEL_SIZE, DESTINATION_PIXEL_SIZE
    ).cpu()

    # ── crops & classification
    crops, masks = masked_patches(labels, rgb)
    if len(crops) == 0:
        return {"coords": np.empty((0, 2)),
                "classes": np.empty((0,)),
                "confidences": np.empty((0, 3))}

    x = torch.cat((crops / 255.0, masks), 1).to("cuda")
    with torch.amp.autocast("cuda"):
        y_hat = classifier(x).cpu()[:, -3:]  # (N,3)

    conf    = y_hat.softmax(1).numpy()
    classes = y_hat.argmax(1).numpy()

    # ── centroid coordinates back to slide space
    cent, _ = safe_centroids(labels)
    coords  = cent[:, (1, 0)].numpy()            # swap to x,y without negative step
    coords *= DESTINATION_PIXEL_SIZE / ORIGINAL_PIXEL_SIZE
    coords += np.array([x1, y1])

    return {"coords": coords, "classes": classes, "confidences": conf}

# ────────────────────────────────────────────────────────────────────
# JSON utilities

def assemble_json(coords, confs):
    def template(name):
        return {"name": name, "type": "Multiple points",
                "version": {"major": 1, "minor": 0}, "points": []}

    outs = {k: template(k) for k in ["lymphocytes", "monocytes", "inflammatory-cells"]}

    for i, (xy, cf) in enumerate(zip(coords, confs)):
        x, y = xy
        base = {"name": f"Point {i}",
                "point": [x * ORIGINAL_PIXEL_SIZE / 1000,
                           y * ORIGINAL_PIXEL_SIZE / 1000,
                           ORIGINAL_PIXEL_SIZE]}
        outs["inflammatory-cells"]["points"].append({**base, "probability": float(cf[0] + cf[1])})
        outs["lymphocytes"]["points"].append({**base, "probability": float(cf[0])})
        outs["monocytes"]["points"].append({**base, "probability": float(cf[1])})
    return outs

# ────────────────────────────────────────────────────────────────────
# Main

def main():
    args = parse_cli()
    if args.bboxes is None or len(args.bboxes) % 4:
        raise ValueError("--bboxes must contain 4×N numbers")

    os.makedirs(args.output_dir, exist_ok=True)
    print("Processing slide:", args.wsi_path)

    instanseg, classifier = load_models(args.model_dir)
    slide = TiffSlide(args.wsi_path)

    all_results = []
    for i in range(0, len(args.bboxes), 4):
        bbox = args.bboxes[i:i+4]
        print(" bbox:", bbox)
        all_results.append(process_bbox(slide, bbox, instanseg, classifier))

    coords      = np.concatenate([r["coords"]      for r in all_results])
    classes     = np.concatenate([r["classes"]     for r in all_results])
    confidences = np.concatenate([r["confidences"] for r in all_results])

    js = assemble_json(coords, confidences)
    for fn, key in [("detected-lymphocytes.json", "lymphocytes"),
                    ("detected-monocytes.json",   "monocytes"),
                    ("detected-inflammatory-cells.json", "inflammatory-cells")]:
        with open(Path(args.output_dir)/fn, "w") as f:
            json.dump(js[key], f, indent=4)
        print("saved", fn)

    print("✓ inference finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
