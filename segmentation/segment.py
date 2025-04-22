import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import cv2
from torch.utils.data import DataLoader
from matplotlib import cm
import tifffile
from tiffslide import TiffSlide

from .utils import (
    extract_patches_from_wsi,
    load_all_patches_in_folder,
    create_visualization,
    load_model,
    get_instance_mask
)
from .config import DEFAULT_SEGMENTATION_MODEL_PATH, NUM_CLASSES, PATCH_SIZE, SEGMENTATION_OUTPUT_DIR


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
    # process a WSI through model
    patch_dir = Path(output_dir) / "patches"
    slide_map_path = patch_dir / "slide_map.json"

    import json
    if slide_map_path.exists():
        logging.info(f"Using existing patches and slide map from: {patch_dir}")
        with open(slide_map_path, "r") as f:
            slide_map = {int(k): tuple(v) for k, v in json.load(f).items()}
        inference_flags_path = patch_dir / "inference_flags.json"
        if inference_flags_path.exists():
            with open(inference_flags_path, "r") as f:
                inference_flags = {int(k): v for k, v in json.load(f).items()}
        else:
            inference_flags = {i: True for i in slide_map.keys()}
    else:
        slide_map, inference_flags = extract_patches_from_wsi(wsi_path, patch_dir)
        with open(patch_dir / "slide_map.json", "w") as f:
            json.dump({str(k): list(v) for k, v in slide_map.items()}, f, indent=2)
        with open(patch_dir / "inference_flags.json", "w") as f:
            json.dump({str(k): bool(v) for k, v in inference_flags.items()}, f, indent=2)

    dataset = load_all_patches_in_folder(patch_dir)
    if len(dataset) == 0:
        raise RuntimeError("No valid tissue patches found.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    valid_indices = [i for i, flag in inference_flags.items() if flag]
    valid_images_iter = iter(loader)

    logging.info(f"{len(valid_indices)} patches selected for inference out of {len(slide_map)} total patches")

    for i in tqdm(range(len(slide_map)), desc="Running inference on patches"):
        if i in valid_indices:
            img = next(valid_images_iter)
            with torch.no_grad():
                outputs = model(img.to(device))
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                predictions.append(probs)
        else:
            dummy = np.zeros((NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            predictions.append(dummy)
    torch.cuda.empty_cache()
    # need 1:1 alignment between preds and masks
    assert len(predictions) == len(slide_map), \
        f"Mismatch: {len(predictions)} predictions vs {len(slide_map)} patches"

    max_x = max(x + w for x, y, w, h in slide_map.values())
    max_y = max(y + h for x, y, w, h in slide_map.values())
    return post_process_predictions(predictions, slide_map, (max_y, max_x))


def save_results(full_mask, wsi_name, output_dir, original_path=None):
    # Save segmentation results and visualizations.
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # save raw mask
    full_mask_npy_path = output_dir / f"{wsi_name}_full_mask.npy"
    np.save(full_mask_npy_path, full_mask)
    
    # save instance masks for each class
    instance_mask_paths = {}
    for class_idx in range(1, NUM_CLASSES):
        instance_mask, num_instances = get_instance_mask(full_mask, tissue_type=class_idx)
        if num_instances == 0:
            continue
        
        print("Saving instance mask for class", class_idx)
        instance_mask_path = output_dir / f"{wsi_name}_full_instance_mask_class{class_idx}.npy"
        np.save(instance_mask_path, instance_mask)
        instance_mask_paths[class_idx] = str(instance_mask_path)
        
        print("Visualizing instance mask for class", class_idx)
        instance_mask_normalized = (instance_mask.astype(np.float32) / instance_mask.max()) if instance_mask.max() > 0 else instance_mask
        cmap = cm.get_cmap('nipy_spectral', instance_mask.max() + 1)
        instance_mask_colored = (cmap(instance_mask_normalized)[:, :, :3] * 255).astype(np.uint8)
        
        print("Saving colored instance mask for class", class_idx)
        instance_png_path = output_dir / f"{wsi_name}_full_instance_mask_class{class_idx}.png"
        Image.fromarray(instance_mask_colored).save(instance_png_path)
        
        print("Saving instance mask as TIFF for class", class_idx)
        downsample_factor = 4
        h, w = instance_mask_colored.shape[:2]
        resized_mask = Image.fromarray(instance_mask_colored).resize(
            (w // downsample_factor, h // downsample_factor), resample=Image.BILINEAR
        )
        instance_colored_tiff_path = output_dir / f"{wsi_name}_full_instance_mask_class{class_idx}.tiff"
        resized_mask = cv2.resize(instance_mask_colored, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_LINEAR)
        tifffile.imwrite(instance_colored_tiff_path, resized_mask, photometric='rgb')



    # Save color-coded mask 
    colour_mask = create_visualization(full_mask)
    full_mask_path = output_dir / f"{wsi_name}_full_mask.png"
    colour_mask.save(full_mask_path)

    # Save overlay if original image is available
    if original_path:
        if str(original_path).lower().endswith(('.png', '.jpg', '.jpeg')):
            original = Image.open(original_path).convert("RGB")
        else:
            slide = TiffSlide(original_path)
            original = slide.get_thumbnail((colour_mask.width, colour_mask.height)).convert("RGB").resize(colour_mask.size)
            full_mask_tiff_path = output_dir / f"{wsi_name}_full_mask.tiff"
            tifffile.imwrite(full_mask_tiff_path, np.array(colour_mask).astype(np.uint8))

        overlay = create_visualization(full_mask, original, alpha=0.4)
        overlay.save(output_dir / f"{wsi_name}_overlay.png")

    return {
        "mask_path": str(full_mask_path),
        "mask_npy_path": str(full_mask_npy_path),
        "instance_mask_paths": instance_mask_paths
    }


def run_segment(in_path, model_path=DEFAULT_SEGMENTATION_MODEL_PATH, output_dir=SEGMENTATION_OUTPUT_DIR):
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

    return save_results(full_mask, wsi_name, output_dir, in_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run segmentation on a WSI or image.")
    parser.add_argument("--input", required=True, help="Path to input image or WSI")
    parser.add_argument("--model_path", default=DEFAULT_SEGMENTATION_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default=SEGMENTATION_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    run_segment(args.input, model_path=args.model_path, output_dir=args.output_dir)
