import os
import torch
import numpy as np
from PIL import Image
import cv2
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import (
    extract_patches_from_wsi,
    load_all_patches_in_folder,
    colour_code_mask,
    load_model,
    overlay_mask_on_image
)
from .config import DEFAULT_SEGMENTATION_MODEL_PATH, NUM_CLASSES, PATCH_SIZE, SEGMENTATION_OUTPUT_DIR


def post_process_predictions(predictions, slide_map, original_shape):
    full_mask = np.zeros(original_shape, dtype=np.uint8)

    for i, pred in enumerate(predictions):
        x, y, w, h = slide_map[i]
        pred_class = np.argmax(pred, axis=0)
        full_mask[y:y+h, x:x+w] = pred_class

    for class_idx in range(1, 5):
        class_mask = (full_mask == class_idx).astype(np.uint8)

        if class_idx == 1:
            kernel = np.ones((3, 3), np.uint8)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        elif class_idx in [2, 3]:
            kernel = np.ones((5, 5), np.uint8)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    cv2.drawContours(class_mask, [contour], 0, 0, -1)
        elif class_idx == 4:
            kernel = np.ones((7, 7), np.uint8)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 200:
                    cv2.drawContours(class_mask, [contour], 0, 0, -1)

        full_mask[class_mask == 1] = class_idx

    return full_mask


def run_segment(in_path, model_path=DEFAULT_SEGMENTATION_MODEL_PATH, output_dir=SEGMENTATION_OUTPUT_DIR):
    import json

    os.makedirs(output_dir, exist_ok=True)

    wsi_name = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(output_dir, wsi_name)
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device=device, weights_only=True)

    full_mask_path = os.path.join(out_path, f"{wsi_name}_full_mask.png")
    full_mask_npy_path = os.path.join(out_path, f"{wsi_name}_full_mask.npy")

    if in_path.endswith(('.svs', '.tif', '.tiff')):
        logging.info(f"Segmenting WSI: {in_path}")
        patch_dir = os.path.join(out_path, "patches")

        slide_map_path = os.path.join(patch_dir, "slide_map.json")
        if os.path.exists(slide_map_path):
            # reuse previous patches
            logging.info(f"Using existing patches and slide map from: {patch_dir}")
            import json
            with open(slide_map_path, "r") as f:
                slide_map = {int(k): tuple(v) for k, v in json.load(f).items()}
            # if also saved inference flags before, load them too:
            inference_flags_path = os.path.join(patch_dir, "inference_flags.json")
            if os.path.exists(inference_flags_path):
                with open(inference_flags_path, "r") as f:
                    inference_flags = {int(k): v for k, v in json.load(f).items()}
            else:
                inference_flags = {i: True for i in slide_map.keys()}  # assume all true if missing
        else:
            slide_map, inference_flags = extract_patches_from_wsi(in_path, patch_dir)
            # optionally save inference_flags
            with open(os.path.join(patch_dir, "slide_map.json"), "w") as f:
                json.dump({str(k): list(v) for k, v in slide_map.items()}, f, indent=2)
            with open(os.path.join(patch_dir, "inference_flags.json"), "w") as f:
                json.dump({str(k): v for k, v in inference_flags.items()}, f, indent=2)

        dataset = load_all_patches_in_folder(patch_dir)
        if len(dataset) == 0:
            raise RuntimeError("No valid tissue patches found.")

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        predictions = []
        valid_indices = [i for i, flag in inference_flags.items() if flag]
        valid_images_iter = iter(loader)

        logging.info(f"{len(valid_indices)} patches selected for inference out of {len(slide_map)} total patches")

        for i in tqdm(range(len(slide_map)), desc="Running Segmentation"):
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

        # need 1:1 alignment between preds and masks
        assert len(predictions) == len(slide_map), \
            f"Mismatch: {len(predictions)} predictions vs {len(slide_map)} patches"

        max_x = max(x + w for x, y, w, h in slide_map.values())
        max_y = max(y + h for x, y, w, h in slide_map.values())
        full_mask = post_process_predictions(predictions, slide_map, (max_y, max_x))

    elif in_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        logging.info(f"Segmenting image: {in_path}")
        img = Image.open(in_path).convert("RGB")
        from .dataset import get_validation_augmentations
        transform = get_validation_augmentations()
        img_np = np.array(img)
        transformed = transform(image=img_np)
        img_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output[0] if isinstance(output, tuple) else output
            full_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    else:
        raise ValueError("Unsupported input format")

    np.save(full_mask_npy_path, full_mask)
    colour_mask = colour_code_mask(full_mask)
    colour_mask.save(full_mask_path)

    # also save overlay if original WSI is not huge
    if in_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        original = Image.open(in_path).convert("RGB")
    else:
        # for WSI, get a thumbnail version
        from tiffslide import TiffSlide
        slide = TiffSlide(in_path)
        original = slide.get_thumbnail((colour_mask.width, colour_mask.height)).convert("RGB")

    overlay = overlay_mask_on_image(original, colour_mask, alpha=0.4)
    overlay.save(os.path.join(out_path, f"{wsi_name}_overlay.png"))

    logging.info(f"Saved segmentation mask to: {full_mask_path}")

    return {
        "mask_path": full_mask_path
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run segmentation on a WSI or image.")
    parser.add_argument("--input", required=True, help="Path to input image or WSI")
    parser.add_argument("--model_path", default=DEFAULT_SEGMENTATION_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default=SEGMENTATION_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    run_segment(args.input, model_path=args.model_path, output_dir=args.output_dir)
