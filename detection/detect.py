import os
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from skimage.measure import label as sklabel
from tqdm import tqdm
from tiffslide import TiffSlide
import tifffile

from instanseg import InstanSeg
from instanseg.utils.pytorch_utils import _to_tensor_float32, torch_fastremap, centroids_from_lab, get_masked_patches
from instanseg.inference_class import _rescale_to_pixel_size
import ttach as tta


def run_inflammatory_cell_detection(wsi_path: str, output_dir: Path, model_path: str) -> np.ndarray:
    wsi_path = Path(wsi_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmentation_dir = output_dir.parent.parent / "segmentation" / wsi_path.stem
    tissue_mask_path = segmentation_dir / f"{wsi_path.stem}_tissue_mask.tif"

    if not tissue_mask_path.exists():
        raise FileNotFoundError(f"Tissue mask not found at: {tissue_mask_path}")

    instanseg_model = Path(model_path) / "instanseg_brightfield_monkey.pt"
    classifier_model_paths = [
        Path(model_path) / "1952372.pt",
        Path(model_path) / "1950672.pt",
        Path(model_path) / "1949389_2.pt"
    ]

    destination_pixel_size = 0.5
    rescale_output = destination_pixel_size != 0.5
    patch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_tta = True

    class ModelEnsemble(torch.nn.Module):
        def __init__(self, model_paths, device, use_tta=False):
            super().__init__()
            self.models = torch.nn.ModuleList([
                self.load_model(path, device, use_tta) for path in model_paths
            ])

        def load_model(self, path, device, use_tta):
            model = torch.jit.load(str(path)).eval().to(device)
            if use_tta:
                transforms = tta.Compose([
                    tta.VerticalFlip(),
                    tta.Rotate90([0, 90, 180, 270])
                ])
                model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
            return model

        def forward(self, x):
            with torch.no_grad():
                preds = [model(x) for model in self.models]
                return torch.mean(torch.stack(preds), dim=0)

    instanseg_script = torch.jit.load(instanseg_model).to(device)
    instanseg = InstanSeg(instanseg_script, verbosity=0)
    classifier = ModelEnsemble(classifier_model_paths, device=device, use_tta=use_tta)

    slide = TiffSlide(str(wsi_path))
    mask_slide = TiffSlide(str(tissue_mask_path))

    mask_full = mask_slide.read_region((0, 0), 0, size=mask_slide.dimensions, as_array=True)
    downsample_factor = 8
    H, W = mask_full.shape[:2]
    mask_thumb = cv2.resize(mask_full, (W // downsample_factor, H // downsample_factor), interpolation=cv2.INTER_NEAREST)
    factor = downsample_factor

    mask_labels = sklabel(mask_thumb > 0)

    all_coords = []
    all_confidences = []
    inflammatory_mask = None
    current_id = 1

    for i in range(1, mask_labels.max() + 1):
        mask = mask_labels == i
        bbox = np.argwhere(mask)
        bbox_min, bbox_max = bbox.min(0), bbox.max(0)
        bbox_scaled = (bbox_min * factor).astype(int), (bbox_max * factor).astype(int)

        x0, y0 = bbox_scaled[0][1], bbox_scaled[0][0]
        x1, y1 = bbox_scaled[1][1], bbox_scaled[1][0]
        width, height = x1 - x0, y1 - y0

        image = slide.read_region((x0, y0), 0, (width, height), as_array=True)
        mask_full = mask_slide.read_region((x0, y0), 0, (width, height), as_array=True)

        H_img, W_img = image.shape[:2]

        # Skip very small ROIs
        if H_img < 128 or W_img < 128:
            print(f"Skipping tiny ROI {i} with size {image.shape}, smaller than 128x128")
            continue

        # Pad if necessary for eval_medium_image
        min_size_for_eval = 512
        pad_h = max(0, min_size_for_eval - H_img)
        pad_w = max(0, min_size_for_eval - W_img)
        if pad_h > 0 or pad_w > 0:
            print(f"Padding ROI {i} from ({H_img}, {W_img}) to ({H_img + pad_h}, {W_img + pad_w})")
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            mask_full = np.pad(mask_full, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

        labels, _ = instanseg.eval_medium_image(
            image,
            pixel_size=0.24199951445730394,
            rescale_output=rescale_output,
            tile_size=1024
        )

        tensor = _rescale_to_pixel_size(_to_tensor_float32(image), 0.24199951445730394, destination_pixel_size).to("cpu")
        mask = _rescale_to_pixel_size(_to_tensor_float32(mask_full), 0.24199951445730394, destination_pixel_size).to("cpu")
        labels = labels.to("cpu") * torch.tensor(mask).bool()
        labels = torch_fastremap(labels)

        crops, masks = get_masked_patches(labels, tensor, patch_size=patch_size)
        x = torch.cat((crops / 255.0, masks), dim=1)

        with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
            with torch.no_grad():
                y_hat = torch.cat([
                    classifier(x[i:i+128].float().to(device)) for i in range(0, len(x), 128)
                ], dim=0).cpu()
                conf = y_hat.softmax(1).numpy()
                y_classes = y_hat.argmax(1).numpy()

        coords = centroids_from_lab(labels.squeeze(0) if labels.ndim == 3 else labels)[0].cpu().numpy()
        coords = coords[:, ::-1] * (destination_pixel_size / 0.24199951445730394)
        coords += np.array([x0, y0])

        is_inflam = np.isin(y_classes, [0, 1])
        coords_inflam = coords[is_inflam]
        conf_inflam = conf[is_inflam]

        all_coords.extend(coords_inflam)
        all_confidences.extend(conf_inflam)

        labels_np = labels.cpu().numpy()
        inflam_indices = np.where(np.isin(y_classes, [0, 1]))[0]
        selected_labels = inflam_indices + 1

        instance_mask = np.zeros_like(labels_np, dtype=np.uint16)

        for lbl in selected_labels:
            instance_mask[labels_np == lbl] = current_id
            current_id += 1

        if inflammatory_mask is None:
            H, W = mask_slide.dimensions
            inflammatory_mask = np.zeros((H, W), dtype=np.uint16)

        H, W = inflammatory_mask.shape

        instance_mask = np.squeeze(instance_mask)
        assert instance_mask.ndim == 2
        h, w = instance_mask.shape

        y1 = min(y0 + h, H)
        x1 = min(x0 + w, W)

        yy = slice(y0, y1)
        xx = slice(x0, x1)

        # Crop instance_mask to match the slice (in case it overflows)
        instance_mask_crop = instance_mask[:(y1 - y0), :(x1 - x0)]

        inflammatory_mask[yy, xx] = np.maximum(inflammatory_mask[yy, xx], instance_mask_crop)


    # Save JSON
    json_path = output_dir / "detected-inflammatory-cells.json"
    json_data = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": []
    }
    for i, ((x, y), conf) in enumerate(zip(all_coords, all_confidences)):
        json_data["points"].append({
            "name": f"Point {i}",
            "point": [x * 0.24199951445730394 / 1000, y * 0.24199951445730394 / 1000, 0.24199951445730394],
            "probability": float(conf[0] + conf[1])
        })

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save instance mask
    npy_path = output_dir / "inflam_cell_instance_mask.npy"
    tif_path = output_dir / "inflam_cell_instance_mask.tiff"
    np.save(npy_path, inflammatory_mask)
    tifffile.imwrite(tif_path, inflammatory_mask.astype(np.uint16))
    print(f"Saved instance mask: {npy_path}, {tif_path}")

    return inflammatory_mask