import os
import h5py
import numpy as np
from PIL import Image
import argparse
import random
import torch

from .config import LABEL_COLOURS, DEFAULT_SEGMENTATION_MODEL_PATH, NUM_CLASSES, SEGMENTATION_OUTPUT_DIR
from .utils import get_instance_mask, is_tissue_patch, create_visualization
from .segment import run_segment


def extract_random_tissue_patches(h5_path, n=5, tissue_threshold=0.05, seed=42):
    tissue_indices = []
    with h5py.File(h5_path, 'r') as f:
        images = f['data']
        for i in range(len(images)):
            if is_tissue_patch(images[i], tissue_threshold):
                tissue_indices.append(i)

        if len(tissue_indices) < n:
            raise RuntimeError(f"Only found {len(tissue_indices)} tissue-rich patches (wanted {n})")

        random.seed(seed)
        selected_indices = random.sample(tissue_indices, n)
        return [(images[i].astype(np.uint8), f['labels'][i].astype(np.uint8), i) for i in selected_indices]


def save_patch_and_masks(img, gt_mask, pred_mask, out_dir, idx):
    patch_dir = os.path.join(out_dir, f"patch_{idx}")
    os.makedirs(patch_dir, exist_ok=True)

    img_pil = Image.fromarray(img)
    
    # create visualizations
    gt_coloured = create_visualization(gt_mask)
    pred_coloured = create_visualization(pred_mask)
    overlay_gt = create_visualization(gt_mask, img_pil, alpha=0.4)
    overlay_pred = create_visualization(pred_mask, img_pil, alpha=0.4)

    # Save base image too
    img_pil.save(os.path.join(patch_dir, "patch.png"))
    gt_coloured.save(os.path.join(patch_dir, "mask_gt.png"))
    pred_coloured.save(os.path.join(patch_dir, "mask_pred.png"))
    overlay_gt.save(os.path.join(patch_dir, "overlay_gt.png"))
    overlay_pred.save(os.path.join(patch_dir, "overlay_pred.png"))

    for class_idx in range(1, NUM_CLASSES):  # skip background
        instance_mask, num_instances = get_instance_mask(pred_mask, tissue_type=class_idx)
        if num_instances == 0:
            continue

        np.save(os.path.join(patch_dir, f"instance_mask_class{class_idx}.npy"), instance_mask)

        # visualize instance mask (random color per instance)
        instance_colored = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
        unique_ids = np.unique(instance_mask)
        for uid in unique_ids:
            if uid == 0:
                continue
            color = tuple(np.random.randint(50, 255, size=3).tolist())
            instance_colored[instance_mask == uid] = color

        Image.fromarray(instance_colored).save(os.path.join(patch_dir, f"instance_mask_class{class_idx}.png"))


def main(h5_path, out_dir, model_path=DEFAULT_SEGMENTATION_MODEL_PATH, threshold=0.05, n=5):
    patches = extract_random_tissue_patches(h5_path, n=n, tissue_threshold=threshold)

    for idx, (img, gt_mask, orig_idx) in enumerate(patches):
        patch_dir = os.path.join(out_dir, f"patch_{idx}")
        os.makedirs(patch_dir, exist_ok=True)

        patch_path = os.path.join(patch_dir, "patch.png")
        Image.fromarray(img).save(patch_path)

        # Run segment on saved patch
        result = run_segment(
            in_path=patch_path,
            model_path=model_path,
            output_dir=patch_dir  # saves result inside this folder
        )

        # Load predicted mask
        pred_npy = result["mask_path"].replace(".png", ".npy")
        pred_mask = np.load(pred_npy)

        save_patch_and_masks(img, gt_mask, pred_mask, out_dir, idx)

    print(f"Saved {n} patch results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample tissue patches from H5 and run segmentation.")
    parser.add_argument("--h5_path", default="data/train_data.h5", help="Path to train_data.h5")
    parser.add_argument("--out_dir", default="test", help="Where to save outputs")
    parser.add_argument("--model_path", default=DEFAULT_SEGMENTATION_MODEL_PATH, help="Segmentation model path")
    parser.add_argument("--threshold", type=float, default=0.05, help="Tissue threshold")
    parser.add_argument("--n", type=int, default=5, help="Number of patches to process")
    args = parser.parse_args()

    main(args.h5_path, args.out_dir, args.model_path, args.threshold, args.n)
