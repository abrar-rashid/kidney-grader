import os
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tiffslide import TiffSlide

MICRONS_PER_PIXEL = 0.24199951445730394

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate downsampled TIFF overlay from inflammatory cell coordinates.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSON or NPY file with coordinates.")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the original WSI file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output TIFF file.")
    return parser.parse_args()

def load_coordinates(input_path):
    # Load coordinates from JSON or NPY file.
    ext = os.path.splitext(input_path)[1]
    if ext == ".json":
        with open(input_path, "r") as f:
            data = json.load(f)
            coords = np.array([[p["point"][0], p["point"][1]] for p in data["points"]])
    elif ext == ".npy":
        coords = np.load(input_path)
    else:
        raise ValueError("Unsupported file format. Use JSON or NPY.")
    return coords

def generate_tiff_overlay(coords, wsi_path, output_path):
    # Create a TIFF overlay with inflammatory cells marked
    slide = TiffSlide(wsi_path)

    pixel_coords = (coords * 1000 / MICRONS_PER_PIXEL).astype(np.int32)

    level = slide.get_best_level_for_downsample(5)
    thumb = slide.read_region((0, 0), level, slide.level_dimensions[level], as_array=True)
    scale_x = slide.level_dimensions[0][0] / thumb.shape[1]
    scale_y = slide.level_dimensions[0][1] / thumb.shape[0]
    overlay = thumb.copy()

    # Adjust coordinates for the thumbnail scale
    for x, y in pixel_coords:
        x_ds, y_ds = int(x / scale_x), int(y / scale_y)
        if 0 <= x_ds < overlay.shape[1] and 0 <= y_ds < overlay.shape[0]:
            cv2.circle(overlay, (x_ds, y_ds), 5, (0, 255, 255), 1)  # dots

    # Save the result as a TIFF file
    output_path_tiff = str(output_path).replace(".tiff", "_overlay.tiff")
    cv2.imwrite(output_path_tiff, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Overlay saved at: {output_path_tiff}")

    # Save the result as a PNG file
    output_path_png = str(output_path).replace(".tiff", "_overlay.png")
    cv2.imwrite(output_path_png, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Overlay saved as PNG at: {output_path_png}")

def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords = load_coordinates(args.input_path)
    print(f"Loaded {len(coords)} inflammatory cell coordinates.")

    output_path = output_dir / "inflammatory_cells_overlay.tiff"
    generate_tiff_overlay(coords, args.wsi_path, output_path)

if __name__ == "__main__":
    main()
