from __future__ import annotations

import time
import math
from typing import Tuple, Dict

import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm

from scipy import ndimage as ndi
from skimage.morphology import  remove_small_objects
from skimage.segmentation import watershed, clear_border
from skimage.filters import gaussian
from skimage.morphology import h_maxima

from .improved_unet import ImprovedUNet
from .config import (
    LABEL_COLOURS,
    PATCH_SIZE,
    NUM_CLASSES,
)

class UnionFind:
    # union-find over hashable keys for merging overlapping instances
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if self.parent.get(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent.get(x, x)

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        # union by rank
        rka = self.rank.get(ra, 0)
        rkb = self.rank.get(rb, 0)
        if rka < rkb:
            self.parent[ra] = rb
        elif rkb < rka:
            self.parent[rb] = ra
        else:
            # tie , promote one
            self.parent[rb] = ra
            self.rank[ra] = rka + 1


def _local_instance_segmentation(
        binary_mask: np.ndarray,
        *,
        sigma: float = 2.0,        # blur strength
        h: float = 2.5,            # h-maxima height
        min_peak_dist: int = 25,   # minimum distance between peaks
        min_area: int = 80,       # discard blobs < this area
) -> np.ndarray:
    # split a single 512×512 patch into instances conservatively
    mask = binary_mask.astype(np.uint8)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.int32)

    # distance transform -> smoothing -> suppress shallow peaks
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    dist = gaussian(dist, sigma=sigma, preserve_range=True)
    dist = dist.astype(np.float32)
    # h-maxima keeps only peaks that are at least "h" taller than their surroundings
    peaks_bin = h_maxima(dist, h=h)

    # get peak coordinates after suppression
    peak_coords = np.column_stack(np.nonzero(peaks_bin))
    # enforce min distance between any two peaks
    if peak_coords.size:
        # downsample the candidate peaks with a simple grid strategy
        from sklearn.neighbors import KDTree
        tree = KDTree(peak_coords)
        keep = np.ones(len(peak_coords), dtype=bool)
        for i, pt in enumerate(peak_coords):
            if not keep[i]:
                continue
            idx = tree.query_radius([pt], r=min_peak_dist, return_distance=False)[0]
            idx = idx[idx > i]               # only later neighbours
            keep[idx] = False
        peak_coords = peak_coords[keep]

    # build marker image
    markers = np.zeros_like(mask, dtype=np.int32)
    if peak_coords.size:
        for i, (r, c) in enumerate(peak_coords, start=1):
            markers[r, c] = i
        markers, _ = ndi.label(markers > 0)
    else:
        markers[mask > 0] = 1

    # watershed and clean-up
    labels = watershed(-dist, markers, mask=mask)
    labels = clear_border(labels)
    if min_area > 0:
        label_vals = np.unique(labels)
        label_vals = label_vals[label_vals > 0]
        if len(label_vals) > 1:
            labels = remove_small_objects(labels, min_size=min_area)

    return labels.astype(np.int32)


def get_instance_mask(
    full_mask: np.ndarray, 
    tissue_type: int,
    patch_size: int = PATCH_SIZE,
    overlap_ratio: float = 0.5,
) -> Tuple[np.ndarray, int]:
    # build a full-slide instance mask of all connected tubules of class = tissue_type
    # returns instance_mask (h, w) array and num_instances count
    #
    # strategy: two-pass algorithm with overlapping patches and union-find merging
    # pass 1: slide over full_mask in overlapping patches (50% overlap by default)
    #   - run local instance segmentation on each patch
    #   - merge overlapping instances with left/top neighbors using union-find
    #   - only keep boundary stripes in memory for next patches to merge against
    # pass 2: assign global ids and write final instance mask to memmap
    #   - flatten union-find to get global id mapping
    #   - recompute local segmentation and write global ids to output memmap
    # memory efficient: never loads more than one patch + small stripes at once

    H, W = full_mask.shape
    stride = int(patch_size * (1.0 - overlap_ratio))
    overlap_px = patch_size - stride

    # build list of top-left coordinates for each patch
    x_coords = []
    x = 0
    while x + patch_size < W:
        x_coords.append(x)
        x += stride
    if W - patch_size >= 0:
        x_coords.append(W - patch_size)
    else:
        x_coords.append(0)

    y_coords = []
    y = 0
    while y + patch_size < H:
        y_coords.append(y)
        y += stride
    if H - patch_size >= 0:
        y_coords.append(H - patch_size)
    else:
        y_coords.append(0)

    # deduplicate
    x_coords = sorted(set(x_coords))
    y_coords = sorted(set(y_coords))

    n_rows = len(y_coords)
    n_cols = len(x_coords)
    num_patches = n_rows * n_cols

    print(f"[INFO] Starting instance labeling for class={tissue_type}")
    print(f"[INFO] Slide: {H}x{W}, Patch size: {patch_size}, Overlap: {overlap_px}px, Total patches: {num_patches}")

    uf = UnionFind()
    prev_right_stripe = None
    prev_bottom_stripes = {}

    # pass 1: local segmentation + merging
    print("PASS 1: Local segmentation + merging...")
    t_start_pass1 = time.time()

    for row_idx, y_tl in enumerate(tqdm(y_coords, desc="Rows (Pass 1)")):
        new_bottom_stripes = {}
        prev_right_stripe = None

        for col_idx, x_tl in enumerate(x_coords):
            patch_idx = row_idx * n_cols + col_idx

            patch_sem = full_mask[y_tl:y_tl + patch_size, x_tl:x_tl + patch_size]
            patch_bin = (patch_sem == tissue_type)

            if not patch_bin.any():
                local_ids = np.zeros((patch_size, patch_size), dtype=np.int32)
            else:
                local_ids = _local_instance_segmentation(patch_bin)
                unique_labels = np.unique(local_ids)
                unique_labels = unique_labels[unique_labels > 0]
                for lbl in unique_labels:
                    key = (patch_idx, int(lbl))
                    uf.parent.setdefault(key, key)
                    uf.rank.setdefault(key, 0)

            # merge with left neighbor
            if col_idx > 0 and prev_right_stripe is not None:
                stripeA, idxA = prev_right_stripe
                stripeB = local_ids[:, :overlap_px]
                a_flat = stripeA.ravel()
                b_flat = stripeB.ravel()
                both_fg = np.logical_and(a_flat > 0, b_flat > 0)
                if both_fg.any():
                    pairs = np.stack([a_flat[both_fg], b_flat[both_fg]], axis=1)
                    pairs = np.unique(pairs, axis=0)
                    for lblA, lblB in pairs:
                        uf.union((idxA, int(lblA)), (patch_idx, int(lblB)))

            # merge with top neighbor
            if row_idx > 0 and col_idx in prev_bottom_stripes:
                stripeA, idxA = prev_bottom_stripes[col_idx]
                stripeB = local_ids[:overlap_px, :]
                a_flat = stripeA.ravel()
                b_flat = stripeB.ravel()
                both_fg = np.logical_and(a_flat > 0, b_flat > 0)
                if both_fg.any():
                    pairs = np.stack([a_flat[both_fg], b_flat[both_fg]], axis=1)
                    pairs = np.unique(pairs, axis=0)
                    for lblA, lblB in pairs:
                        uf.union((idxA, int(lblA)), (patch_idx, int(lblB)))

            # cache right stripe for next column
            right_stripe = local_ids[:, patch_size - overlap_px:] if patch_size > overlap_px else local_ids
            prev_right_stripe = (right_stripe.copy(), patch_idx)

            # cache bottom stripe for next row
            bottom_stripe = local_ids[patch_size - overlap_px:, :] if patch_size > overlap_px else local_ids
            new_bottom_stripes[col_idx] = (bottom_stripe.copy(), patch_idx)

        prev_bottom_stripes = new_bottom_stripes

    print(f"PASS 1 complete in {time.time() - t_start_pass1:.1f} sec")

    # assign global instance ids
    print("Assigning global IDs...")
    root_to_global = {}
    next_global_id = 1
    for key in uf.parent.keys():
        root = uf.find(key)
        if root not in root_to_global:
            root_to_global[root] = next_global_id
            next_global_id += 1

    num_instances = next_global_id - 1
    print(f"Total unique instances: {num_instances}")

    dtype = np.uint16 if num_instances < 65535 else np.uint32

    # allocate memmap for full instance mask
    instance_mask = np.memmap(
        filename="instance_mask_memmap.dat",
        dtype=dtype,
        mode="w+",
        shape=(H, W),
    )
    instance_mask[:] = 0

    # pass 2 - write global ids into final mask
    print("PASS 2: Writing global instance IDs...")
    t_start_pass2 = time.time()

    for row_idx, y_tl in enumerate(tqdm(y_coords, desc="Rows (Pass 2)")):
        for col_idx, x_tl in enumerate(x_coords):
            patch_idx = row_idx * n_cols + col_idx

            patch_sem = full_mask[y_tl:y_tl + patch_size, x_tl:x_tl + patch_size]
            patch_bin = (patch_sem == tissue_type)

            if not patch_bin.any():
                continue

            local_ids = _local_instance_segmentation(patch_bin)
            ys, xs = np.nonzero(local_ids)
            for yy, xx in zip(ys, xs):
                local_label = int(local_ids[yy, xx])
                key = (patch_idx, local_label)
                root = uf.find(key)
                global_id = root_to_global[root]
                instance_mask[y_tl + yy, x_tl + xx] = global_id

    print(f"PASS 2 complete in {time.time() - t_start_pass2:.1f} sec")
    print("Instance labeling complete")

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
