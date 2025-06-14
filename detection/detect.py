from pathlib import Path
import subprocess
import json
import numpy as np

from detection.patch_extractor import extract_patches_from_wsi

def run_inflammatory_cell_detection(wsi_path: str, output_dir: Path, model_path: str) -> np.ndarray:
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

    return coords
