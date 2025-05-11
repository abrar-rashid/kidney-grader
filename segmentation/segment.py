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


def post_process_predictions(predictions, slide_map, original_shape):
    full_mask = np.zeros(original_shape, dtype=np.uint8)
    for i, pred in enumerate(predictions):
        x, y, w, h = slide_map[i]
        pred_class = np.argmax(pred, axis=0)
        full_mask[y:y+h, x:x+w] = pred_class
    return full_mask


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
    return full_mask

def save_results(full_mask, wsi_name, output_dir, original_path=None, visualise=False):
    # Save segmentation results and visualizations.

    def downsample_array(arr, factor):
        h, w = arr.shape
        new_width = int(w / factor)
        new_height = int(h / factor)
        return cv2.resize(arr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

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
        if class_idx in [2, 3]:
            continue # skip veins and arteries for now

        instance_mask, num_instances = get_instance_mask(full_mask, tissue_type=class_idx)
        if num_instances == 0:
            continue

        print(f"Saving instance mask for class {class_idx}")
        instance_mask_path = output_dir / f"{wsi_name}_full_instance_mask_class{class_idx}.npy"
        np.save(instance_mask_path, instance_mask)
        result["instance_mask_paths"][class_idx] = str(instance_mask_path)

        if visualise:
            print(f"Visualizing instance mask for class {class_idx}")
            small_mask = downsample_array(instance_mask, factor=4)
            visualize_instance_mask(small_mask, output_dir / f"{wsi_name}_full_instance_mask_class{class_idx}.tiff")

        del instance_mask
        gc.collect()

    # clean up memmap if it exists
    memmap_path = output_dir / "full_mask_memmap.dat"
    if memmap_path.exists():
        try:
            memmap_path.unlink()
            print(f"Deleted memmap file: {memmap_path}")
        except Exception as e:
            print(f"Warning: Could not delete memmap file: {e}")

    if visualise:
        print("Creating visualization of semantic mask")
        colour_mask = create_visualization(full_mask)
        semantic_mask_tiff_path = output_dir / f"{wsi_name}_semantic_mask_colored.tiff"
        tifffile.imwrite(semantic_mask_tiff_path, np.array(colour_mask), photometric='rgb')
        result["semantic_mask_colored_tiff_path"] = str(semantic_mask_tiff_path)

        # Optionally save raw semantic mask for debugging
        semantic_mask_npy_path = output_dir / f"{wsi_name}_semantic_mask.npy"
        np.save(semantic_mask_npy_path, np.array(full_mask))
        result["semantic_mask_npy_path"] = str(semantic_mask_npy_path)

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


def run_segment(in_path, model_path=DEFAULT_SEGMENTATION_MODEL_PATH,
                output_dir=SEGMENTATION_OUTPUT_DIR, visualise=False):
    # Entry function to run segmentation on WSI or regular image.
    wsi_name = Path(in_path).stem
    output_dir = Path(output_dir) / wsi_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device=device, weights_only=True)

    if str(in_path).lower().endswith(('.svs', '.tif', '.tiff')):
        logging.info(f"Segmenting WSI: {in_path}")
        full_mask = process_wsi(in_path, model, device, output_dir)
    elif str(in_path).lower().endswith(('.png', '.jpg', '.jpeg')):
        logging.info(f"Segmenting image: {in_path}")
        full_mask = process_regular_image(in_path, model, device)
    else:
        raise ValueError("Unsupported input format")

    return save_results(full_mask, wsi_name, output_dir, in_path, visualise)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run segmentation on a WSI or image.")
    parser.add_argument("--input", required=True, help="Path to input image or WSI")
    parser.add_argument("--model_path", default=DEFAULT_SEGMENTATION_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default=SEGMENTATION_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    run_segment(args.input, model_path=args.model_path, output_dir=args.output_dir)
