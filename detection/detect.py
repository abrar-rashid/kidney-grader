# detection/instseg_infer.py

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import tifffile
from tiffslide import TiffSlide
from pathlib import Path
from instanseg import InstanSeg
from skimage.measure import label as sklabel
from instanseg.utils.pytorch_utils import torch_fastremap, centroids_from_lab, get_masked_patches, _to_tensor_float32
from instanseg.inference_class import _rescale_to_pixel_size

# Setup
MODEL_DIR = "detection/models"
INSTANSEG_MODEL = "instanseg_brightfield_monkey.pt"
CLASSIFIER_MODELS = ["1952372.pt", "1950672.pt", "1949389_2.pt"]
DEST_PIXEL_SIZE = 0.5
PATCH_SIZE = 128


class InflammatoryCellDetector:
    def __init__(self, instanseg_model_path, classifier_model_paths, device="cuda"):
        self.device = device
        self.inst_model = torch.jit.load(instanseg_model_path).to(device).eval()
        self.detector = InstanSeg(self.inst_model, verbosity=0)

        self.classifier = torch.nn.ModuleList([
            torch.jit.load(p).to(device).eval() for p in classifier_model_paths
        ])

    def classify_batch(self, x, batch_size=128):
        preds_all = []

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in range(0, len(x), batch_size):
                xb = x[i:i+batch_size].float()
                preds = [m(xb) for m in self.classifier]
                pred = torch.stack(preds).mean(0)
                preds_all.append(pred)

        y_hat = torch.cat(preds_all, dim=0)
        return y_hat.softmax(1).argmax(1)

    def detect(self, img_array, mask_array):
        img_tensor = _to_tensor_float32(img_array).to(self.device)
        mask_tensor = _to_tensor_float32(mask_array).to(self.device)

        labels, _ = self.detector.eval_medium_image(
            img_array, pixel_size=0.242, rescale_output=True, seed_threshold=0.1, tile_size=1024
        )
        labels = labels.to(self.device) * torch.tensor(mask_tensor).bool()
        labels = torch_fastremap(labels)
        crops, masks = get_masked_patches(labels, img_tensor, patch_size=PATCH_SIZE)
        x = torch.cat((crops / 255.0, masks), dim=1).to(self.device)
        
        torch.cuda.empty_cache()
        y_hat = self.classify_batch(x)
        coords = centroids_from_lab(labels)[0].cpu().numpy()

        inflam_cell_coords = coords[y_hat.cpu().numpy() == 1]  # class 1 = Inflammatory Cell

        return inflam_cell_coords, labels, y_hat


def run_inflammatory_cell_detection(wsi_path: str, output_dir: Path, model_path: str) -> np.ndarray:
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide = TiffSlide(wsi_path)
    image = slide.read_region((0, 0), level=0, size=slide.dimensions, as_array=True)
    mask = np.ones_like(image[..., 0], dtype=np.uint8)  # Use full image as dummy mask

    detector = InflammatoryCellDetector(
        instanseg_model_path=os.path.join(model_path, INSTANSEG_MODEL),
        classifier_model_paths=[os.path.join(model_path, ckpt) for ckpt in CLASSIFIER_MODELS],
    )

    coords, _, _ = detector.detect(image, mask)

    # create instance mask with unique labels
    inflam_cell_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    for idx, (y, x) in enumerate(coords.astype(int)):
        if 0 <= y < inflam_cell_mask.shape[0] and 0 <= x < inflam_cell_mask.shape[1]:
            inflam_cell_mask[y, x] = idx + 1

    np.save(output_dir / "inflam_cell_instance_mask.npy", inflam_cell_mask)

    # Create RGB colormapped mask
    normalized_mask = (inflam_cell_mask.astype(np.float32) / inflam_cell_mask.max()) if inflam_cell_mask.max() > 0 else inflam_cell_mask
    rgb_mask = (plt.cm.tab20(normalized_mask)[..., :3] * 255).astype(np.uint8)
    tifffile.imwrite(output_dir / "inflam_cell_instance_mask.tiff", rgb_mask, photometric="rgb")

    # Overlay points on WSI (green dots)
    overlay_points = image[..., :3].copy()
    if overlay_points.dtype != np.uint8:
        overlay_points = (overlay_points / overlay_points.max() * 255).astype(np.uint8)

    for (y, x) in coords.astype(int):
        cv2.circle(overlay_points, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

    tifffile.imwrite(output_dir / "wsi_with_inflam_overlay.tiff", overlay_points, photometric="rgb")

    # Optional: Blended version of RGB mask + WSI
    alpha = 0.5
    blended = cv2.addWeighted(overlay_points, 1 - alpha, rgb_mask, alpha, 0)
    tifffile.imwrite(output_dir / "wsi_with_mask_blend.tiff", blended, photometric="rgb")

    return inflam_cell_mask

