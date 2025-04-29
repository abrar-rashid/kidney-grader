# Credit to Vishal Jain, as this file has been adapted from his inference.py.

import torch
import cv2
import numpy as np
from typing import List
from instanseg import InstanSeg
from instanseg.utils.pytorch_utils import (
    _to_tensor_float32,
    get_masked_patches,
    centroids_from_lab,
    torch_fastremap,
)
from instanseg.inference_class import _rescale_to_pixel_size
import ttach as tta

DEST_PIXEL_SIZE = 0.5
PATCH_SIZE = 128

class ModelEnsemble(torch.nn.Module):
    def __init__(self, model_paths: List[str], device: str = "cuda", use_tta: bool = True):
        super().__init__()
        self.models = torch.nn.ModuleList([
            self.load_model(p, device, use_tta) for p in model_paths
        ])
        self.device = device

    def load_model(self, path: str, device: str, use_tta: bool):
        model = torch.jit.load(path).eval().to(device)
        if use_tta:
            transforms = tta.Compose([
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ])
            model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
        return model

    def forward(self, x):
        with torch.no_grad():
            preds = [model(x) for model in self.models]
            return torch.mean(torch.stack(preds), dim=0)


def detect_cells_in_image(
    image: np.ndarray,
    mask: np.ndarray,
    instanseg_model_path: str,
    classifier_model_paths: List[str],
    patch_size: int = 128,
    destination_pixel_size: float = 0.5,
    pixel_size_original: float = 0.24199951445730394
):
    device = "cuda"
    classification_device = "cuda"

    instanseg_model = torch.jit.load(instanseg_model_path).to(device)
    detector = InstanSeg(instanseg_model, verbosity=0)
    classifier = ModelEnsemble(classifier_model_paths, device=classification_device, use_tta=True)

    labels, _ = detector.eval_medium_image(
        image,
        pixel_size=pixel_size_original,
        rescale_output=(destination_pixel_size != pixel_size_original),
        seed_threshold=0.1,
        tile_size=1024
    )

    tensor = _rescale_to_pixel_size(_to_tensor_float32(image), pixel_size_original, destination_pixel_size).to(device)
    labels = labels.to(device)

    if labels.ndim == 4 and labels.shape[0] == 1 and labels.shape[1] == 1:
        labels = labels.squeeze(0).squeeze(0)
    elif labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels.squeeze(0)
    elif labels.ndim != 2:
        raise ValueError(f"[ERROR] Unexpected label shape: {labels.shape}")

    if mask.shape != labels.shape[-2:]:
        mask = cv2.resize(mask.astype(np.uint8), (labels.shape[-1], labels.shape[-2]), interpolation=cv2.INTER_NEAREST)
    labels *= torch.tensor(mask).bool().to(device)

    if labels.max() == 0:
        return np.empty((0, 2)), np.array([]), np.empty((0, 3))

    labels = torch_fastremap(labels)

    crops, masks = get_masked_patches(labels, tensor, patch_size)
    x = torch.cat((crops / 255.0, masks), dim=1)

    with torch.amp.autocast("cuda"):
        with torch.no_grad():
            batch_size = 128
            y_hat = torch.cat([classifier(x[i:i + batch_size].float().to(classification_device))
                               for i in range(0, len(x), batch_size)], dim=0)
            y_hat = y_hat[:, -3:].cpu()

    coords = centroids_from_lab(labels)[0].cpu().numpy()
    classes = y_hat.argmax(1).numpy()
    confidences = y_hat.softmax(1).numpy()

    inflam_mask = (classes == 0) | (classes == 1)

    print("[DEBUG] Class histogram:", np.bincount(classes))

    coords = coords[inflam_mask]
    classes = classes[inflam_mask]
    confidences = confidences[inflam_mask]

    return coords, classes, confidences