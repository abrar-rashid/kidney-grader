from __future__ import annotations
import gc
import logging
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import torch
import cv2
from torch.utils.data import DataLoader
from matplotlib import cm
import tifffile
from tiffslide import TiffSlide
from .utils import (
    create_visualization,
    load_model,
    get_instance_mask
)
from detection.patch_extractor import extract_patches_from_wsi
from .config import DEFAULT_SEGMENTATION_MODEL_PATH, NUM_CLASSES, PATCH_OVERLAP, PATCH_SIZE, SEGMENTATION_OUTPUT_DIR


def process_regular_image(img_path, model, device):
    # process a patch/non-wsi image through the model
    img = Image.open(img_path).convert("RGB")
    from .dataset import get_validation_augmentations
    transform = get_validation_augmentations()
    img_np = np.array(img)
    transformed = transform(image=img_np)
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        output = output[0] if isinstance(output, tuple) else output
        return torch.argmax(output.squeeze(), dim=0).cpu().numpy()


def process_wsi(wsi_path, model, device, output_dir):
    from torchvision import transforms

    logging.info(f"Extracting patches from WSI: {wsi_path}")
    patches = extract_patches_from_wsi(
        wsi_path=wsi_path,
        patch_size=PATCH_SIZE,
        overlap=PATCH_OVERLAP,
        level=0,
        tissue_threshold=0.05,
        create_debug_images=False,
        debug_output_dir=None,
        num_patches=float("inf"),
        extraction_mode="contiguous",
        save_patches=False,
        output_dir=None,
        label=None,
    )

    if not patches:
        raise RuntimeError("No valid tissue patches found for inference.")

    logging.info(f"{len(patches)} tissue patches extracted for inference.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8, 0.65, 0.8], std=[0.15, 0.15, 0.15])
    ])

    coords = [(x, y, PATCH_SIZE, PATCH_SIZE) for _, x, y in patches]
    max_x = max(x + w for x, y, w, h in coords)
    max_y = max(y + h for x, y, w, h in coords)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    memmap_path = Path(output_dir) / "full_mask_memmap.dat"

    try:
        full_mask = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(max_y, max_x))
        full_mask[:] = 0

        for j, (patch_np, x, y) in tqdm(enumerate(patches), total=len(patches), desc="Running inference on patches"):
            img_tensor = transform(Image.fromarray(patch_np)).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                pred_class = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy()
                full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_class

            if j % 500 == 0:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        
        # convert memmap to regular array to avoid file handle issues
        full_mask_array = np.array(full_mask)
        
        # explicitly delete memmap reference and clean up
        del full_mask
        
        return full_mask_array
        
    finally:
        # always cleanup memmap file, even if processing fails
        if memmap_path.exists():
            try:
                memmap_path.unlink()
                print(f"Cleaned up memmap file: {memmap_path}")
            except Exception as e:
                print(f"Warning: Could not delete memmap file: {e}")


def process_regular_image_tensor(img_tensor, model, device):
    # for eval script, which uses tensors from the h5 test dataset
    if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)  # Remove batch dimension if size 1

    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        output = output[0] if isinstance(output, tuple) else output
        return torch.argmax(output.squeeze(), dim=0).cpu().numpy()


def visualize_instance_mask(mask: np.ndarray, output_path: Path, cmap_name: str = "nipy_spectral") -> None:
    norm_mask = mask.astype(np.float32) / mask.max() if mask.max() > 0 else mask
    cmap = cm.get_cmap(cmap_name, int(mask.max()) + 1)
    color_image = (cmap(norm_mask)[..., :3] * 255).astype(np.uint8)
    tifffile.imwrite(str(output_path), color_image, photometric="rgb")


from typing import Optional

import numpy as np
from matplotlib import cm
from PIL import Image


def colourise_instances(
    mask: np.ndarray,
    cmap_name: str = "nipy_spectral",
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Image.Image:
    # credit to ChatGPT. Colours instances in the mask and ensures neighbouring instances are non-similar colours

    """
    Map each *instance id* in a 2‑D label image to a unique RGB colour.

    * **Vectorised:** O(N) time, no Python loops over instance ids.
    * **Deterministic option:** pass `shuffle=False` or a fixed `seed`.
    * Works with NumPy arrays *and* mem‑maps; the input is never copied.

    Parameters
    ----------
    mask : np.ndarray
        2‑D array where background = 0 and each connected component has a
        positive integer id (output of `get_instance_mask`).
    cmap_name : str, optional
        Any Matplotlib colormap name (default `"nipy_spectral"`).
    shuffle : bool, optional
        If *True* (default) colours are randomly permuted so neighbouring
        ids receive very different hues.  Set to *False* for reproducible
        yet still visually distinct colouring.
    seed : int or None, optional
        Seed for the RNG that does the shuffling.  Ignored when
        `shuffle=False`.

    Returns
    -------
    PIL.Image.Image
        RGB image (mode `"RGB"`) the same H×W as `mask`.
    """
    if mask.ndim != 2:
        raise ValueError("`mask` must be a 2‑D array")

    # ------------------------------------------------------------------ LUT
    max_id = int(mask.max())
    if max_id == 0:                     # nothing but background
        return Image.fromarray(np.zeros((*mask.shape, 3), np.uint8), mode="RGB")

    # Build colour look‑up table once, length = (max_id + 1)
    cmap = cm.get_cmap(cmap_name, max_id + 1)
    lut = (cmap(np.arange(max_id + 1))[:, :3] * 255).astype(np.uint8)

    # Optional permutation so adjacent ids → distant colours
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(max_id) + 1   # exclude index 0 (background)
        lut[1:] = lut[perm]

    # ------------------------------------------------------------------ map
    # NumPy advanced indexing turns the 2‑D label image straight into an
    # H×W×3 RGB array in one vectorised step, executed in C.
    rgb = lut[mask]

    return Image.fromarray(rgb, mode="RGB")

def overlay_rgb(base: Image.Image, mask_rgb: Image.Image, alpha: float = 0.4) -> Image.Image:
    return Image.blend(base, mask_rgb, alpha)

def save_results(full_mask, wsi_name, output_dir, original_path=None, visualise=False):
    # Save segmentation results and visualizations.

    def downsample_array(arr, factor, copy: bool = False):
        if factor < 1:
            raise ValueError("factor must be a positive integer")

        # Trim the edges so we take whole blocks only (matches cv2.resize).
        h, w   = arr.shape[:2]
        h_trim = (h // factor) * factor
        w_trim = (w // factor) * factor

        # Strided nearest-neighbour pick – *no allocation* here.
        ds_view = arr[:h_trim:factor, :w_trim:factor]

        return ds_view.copy() if copy else ds_view

    def visualize_instance_mask(mask, output_path, cmap_name='nipy_spectral'):
        norm_mask = mask.astype(np.float32) / mask.max() if mask.max() > 0 else mask
        cmap = cm.get_cmap(cmap_name, int(mask.max()) + 1)
        color_image = (cmap(norm_mask)[..., :3] * 255).astype(np.uint8)
        tifffile.imwrite(output_path, color_image, photometric='rgb')

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    result = {"instance_mask_paths": {}}

    # save instance masks for each class
    for class_idx in range(1, NUM_CLASSES):
        if class_idx in [2, 3, 4]:
            continue # skip veins and arteries for now

        instance_mask, num_instances = get_instance_mask(full_mask, tissue_type=class_idx)
        print(f"Class {class_idx}: {num_instances} instances found")
        if num_instances == 0:
            continue

        print(f"Saving instance mask for class {class_idx}")
        # determine the optimal data type based on the maximum ID
        max_id = instance_mask.max()
        dtype = np.uint16 if max_id < 65535 else np.uint32
        instance_mask = instance_mask.astype(dtype, copy=False)

        # construct the output file path as a BigTIFF file
        tiff_path = output_dir / f"{wsi_name}_full_instance_mask_class{class_idx}.tiff"

        # save the instance mask as a compressd tiled BigTIFF
        tifffile.imwrite(
            str(tiff_path),
            instance_mask,
            bigtiff=True,
            compression=("zstd", 5),
            tile=(512, 512),
            photometric='minisblack'
        )

        # update the result dictionary with the new file path
        result["instance_mask_paths"][class_idx] = str(tiff_path)

        if visualise:
            # print(f"Visualizing instance mask for class {class_idx}")
            # small_mask = downsample_array(instance_mask, factor=4)
            # visualize_instance_mask(small_mask, output_dir / f"{wsi_name}_full_instance_mask_class{class_idx}.tiff")

            if original_path is not None:
                try:
                    if str(original_path).lower().endswith((".png", ".jpg", ".jpeg")):
                        original = Image.open(original_path).convert("RGB")
                    else:
                        slide = TiffSlide(original_path)
                        original = slide.read_region((0, 0), 0, (full_mask.shape[1], full_mask.shape[0])).convert("RGB")

                    # down‑sample both mask and image same as semantic mask
                    downsample_factor = 8
                    small_inst_mask = downsample_array(instance_mask, downsample_factor, copy=True)

                    instance_mask.flush()
                    del instance_mask
                    gc.collect()

                    small_original = original.resize((small_inst_mask.shape[1], small_inst_mask.shape[0]))
                    coloured_mask = colourise_instances(small_inst_mask)
                    overlay_img = overlay_rgb(small_original, coloured_mask, alpha=0.4)

                    overlay_path = output_dir / f"{wsi_name}_overlay_class{class_idx}.png"
                    overlay_img.save(overlay_path)
                    result["instance_overlay_paths"][class_idx] = str(overlay_path)
                    print(f"Instance overlay saved: {overlay_path}")
                except Exception as e:
                    print(f"Warning: Instance overlay for class {class_idx} failed: {e}")

    if visualise:
        print("Creating visualization of semantic mask")
        colour_mask = create_visualization(full_mask)

        if original_path:
            try:
                print("Saving overlay visualization")
                print(f"Original image path: {original_path}")
                if str(original_path).lower().endswith(('.png', '.jpg', '.jpeg')):
                    original = Image.open(original_path).convert("RGB")
                else:
                    slide = TiffSlide(original_path)
                    colour_mask_size = (colour_mask.width, colour_mask.height)
                    original = slide.read_region((0, 0), 0, colour_mask_size).convert("RGB")

                # downsample visualization for optimization
                downsample_factor = 8
                small_mask = downsample_array(full_mask, downsample_factor)

                # resize original to match downsampled mask size
                small_original = original.resize((small_mask.shape[1], small_mask.shape[0]))
                if small_original.size != (small_mask.shape[1], small_mask.shape[0]):
                    print(f"Warning: Size mismatch between mask and original image: "
                          f"mask={small_mask.shape}, original={small_original.size}")

                # mask overlaid on wsi
                overlay = create_visualization(small_mask, small_original, alpha=0.4)
                overlay_path = output_dir / f"{wsi_name}_overlay.png"
                overlay.save(overlay_path)
                print(f"Overlay saved at: {overlay_path}")
                result["overlay_path"] = str(overlay_path)

            except Exception as e:
                print(f"Warning: Overlay visualization failed due to: {e}")

    # clean up resources
    del full_mask
    gc.collect()

    return result


def run_segment(in_data, model_path=DEFAULT_SEGMENTATION_MODEL_PATH,
                output_dir=SEGMENTATION_OUTPUT_DIR, visualise=False):
    # Entry function to run segmentation on WSI or regular image.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device=device, weights_only=True)

    # check if input is a file path or a tensor
    if isinstance(in_data, str):
        wsi_name = Path(in_data).stem
        output_dir = Path(output_dir)

        if in_data.lower().endswith(('.svs', '.tif', '.tiff')):
            logging.info(f"Segmenting WSI: {in_data}")
            full_mask = process_wsi(in_data, model, device, output_dir)
            return save_results(full_mask, wsi_name, output_dir, original_path=in_data, visualise=visualise)
        elif in_data.lower().endswith(('.png', '.jpg', '.jpeg')):
            logging.info(f"Segmenting image: {in_data}")
            full_mask = process_regular_image(in_data, model, device)
            return save_results(full_mask, wsi_name, output_dir, original_path=in_data, visualise=visualise)
        else:
            raise ValueError("Unsupported input format")
    elif torch.is_tensor(in_data):
        logging.info(f"Segmenting tensor image")
        full_mask = process_regular_image_tensor(in_data, model, device)
        return full_mask
    else:
        raise ValueError("Unsupported input type")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run segmentation on a WSI or image.")
    parser.add_argument("--input", required=True, help="Path to input image or WSI")
    parser.add_argument("--model_path", default=DEFAULT_SEGMENTATION_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default=SEGMENTATION_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    run_segment(args.input, model_path=args.model_path, output_dir=args.output_dir)
