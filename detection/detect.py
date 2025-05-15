from pathlib import Path
import subprocess
import tempfile
import json
import cv2
from tiffslide import TiffSlide
import copy

import numpy as np

from detection.patch_extractor import extract_patches_from_wsi

def run_inflammatory_cell_detection(wsi_path: str, output_dir: Path, model_path: str, visualise: bool = False) -> np.ndarray:
    MICRONS_PER_PIXEL = 0.24199951445730394

    patches = extract_patches_from_wsi(
        wsi_path=wsi_path,
        patch_size=2048,
        overlap=0,
        level=0,
        tissue_threshold=0.05,
        create_debug_images=False,
        debug_output_dir="./tmp/debug",
        num_patches=float("inf"),
        exclusion_conditions=[],
        exclusion_mode="any",
        extraction_mode="contiguous",
        save_patches=False,
        output_dir="./tmp/patches",
        label=None,
    )

    bbox_list = []
    for patch_np, x, y in patches:
        if patch_np is not None and patch_np.shape[0] > 0 and patch_np.shape[1] > 0:
            ymin, xmin = y, x
            ymax, xmax = y + patch_np.shape[0], x + patch_np.shape[1]
            bbox_list.extend([ymin, xmin, ymax, xmax])

    if not bbox_list:
        raise RuntimeError("No valid tissue regions found for inference.")

    command = [
        "python3", "detection/inference.py",
        "--wsi_path", wsi_path,
        "--output_dir", str(output_dir),
        "--model_dir", str(model_path),
        "--bbox", *map(str, bbox_list)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during inference: {result.stderr}")
        return np.array([])

    inflammatory_cells_path = output_dir / "detected-inflammatory-cells.json"
    with open(inflammatory_cells_path, "r") as f:
        inflammatory_cells = json.load(f)

    coords = np.array([[p["point"][0], p["point"][1]] for p in inflammatory_cells["points"]])

    threshold_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]       # probabilities to keep
    for thr in threshold_list:
        # keep only points whose probability ≥ thr
        filtered_points = [
            pt for pt in inflammatory_cells["points"]
            if pt.get("probability", 0.0) >= thr
        ]

        # skip empty results to avoid writing empty files
        if not filtered_points:
            continue

        filtered_dict = copy.deepcopy(inflammatory_cells)
        filtered_dict["points"] = filtered_points

        # build filename like: detected-inflammatory-cells-p02.json
        thr_tag = f"{thr:.2f}".replace(".", "") # 0.2 is “02”, 0.3 is “03”, etc
        out_path = output_dir / f"detected-inflammatory-cells-p{thr_tag}.json"

        with open(out_path, "w") as f_out:
            json.dump(filtered_dict, f_out, indent=4)

        print(f"Saved filtered inflammatory cells (p ≥ {thr}) to {out_path}")

    if visualise:
        slide = TiffSlide(wsi_path)

        # downsample factor of 5x is the highest resolution that qupath can handle
        level = slide.get_best_level_for_downsample(5)
        thumb = slide.read_region((0, 0), level, slide.level_dimensions[level], as_array=True)

        scale_x = slide.level_dimensions[0][0] / thumb.shape[1]
        scale_y = slide.level_dimensions[0][1] / thumb.shape[0]

        overlay = thumb.copy()

        for x, y in coords:
            x_ds, y_ds = int(x / scale_x), int(y / scale_y)
            if 0 <= x_ds < overlay.shape[1] and 0 <= y_ds < overlay.shape[0]:
                cv2.circle(overlay, (x_ds, y_ds), 3, (0, 255, 255), -1)

        # save as PNG
        output_image_path_png = output_dir / "inflammatory_cells_overlay_downsampled.png"
        cv2.imwrite(str(output_image_path_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Overlay saved as PNG at: {output_image_path_png}")

        # save as TIFF
        output_image_path_tiff = output_dir / "inflammatory_cells_overlay_downsampled.tiff"
        cv2.imwrite(str(output_image_path_tiff), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Overlay saved as TIFF at: {output_image_path_tiff}")

    return coords
