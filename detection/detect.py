from pathlib import Path
import subprocess
import tempfile
import json
import cv2
from tiffslide import TiffSlide

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
        num_patches=1000,
        exclusion_conditions=[],
        exclusion_mode="any",
        extraction_mode="contiguous",
        save_patches=False,
        output_dir="./tmp/patches",
        label=None,
    )

    bbox_list = []
    for patch_np, x, y in patches:
        x1, y1 = x, y
        x2, y2 = x + patch_np.shape[1], y + patch_np.shape[0]
        bbox_list.extend([x1, y1, x2, y2])

    if not bbox_list:
        raise RuntimeError("No valid tissue regions found for inference.")

    command = [
        "python3", "detection/inference.py",
        "--wsi_path", wsi_path,
        "--output_dir", str(output_dir),
        "--model_dir", str(model_path),
        "--bboxes", *map(str, bbox_list),
    ]

    print("Running inference script...")
    subprocess.run(command, check=True)

    inflammatory_json = output_dir / "detected-inflammatory-cells.json"
    with open(inflammatory_json) as f:
        data = json.load(f)

    points = np.array([point["point"][:2] for point in data["points"]])
    np.save(output_dir / "inflam_cell_mm_coords.npy", points)

    if visualise:
        slide = TiffSlide(wsi_path)
        pixel_coords = (points * 1000 / MICRONS_PER_PIXEL).astype(np.int32)
        np.save(output_dir / "inflam_cell_pixel_coords.npy", pixel_coords)

        level = slide.get_best_level_for_downsample(32)
        thumb = slide.read_region((0, 0), level, slide.level_dimensions[level], as_array=True)
        scale_x = slide.level_dimensions[0][0] / thumb.shape[1]
        scale_y = slide.level_dimensions[0][1] / thumb.shape[0]
        overlay = thumb.copy()

        for x, y in pixel_coords:
            x_ds, y_ds = int(x / scale_x), int(y / scale_y)
            if 0 <= x_ds < overlay.shape[1] and 0 <= y_ds < overlay.shape[0]:
                cv2.circle(overlay, (x_ds, y_ds), 3, (0, 0, 255), -1)

        cv2.imwrite(str(output_dir / "inflam_overlay.tiff"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # for QuPath
        with open(output_dir / "inflammatory_cells_qupath.tsv", "w") as f:
            f.write("Name\tX\tY\tClass\n")
            for i, (x, y) in enumerate(pixel_coords):
                f.write(f"Point {i}\t{x}\t{y}\tinflam\n")

    return points