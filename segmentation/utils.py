import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from tiffslide import TiffSlide
from torchvision import transforms
import torch
from .improved_unet import ImprovedUNet
from .config import LABEL_COLOURS, PATCH_SIZE, PATCH_OVERLAP, TISSUE_THRESHOLD, PATCH_LEVEL, NUM_CLASSES


def colour_code_mask(mask):
    h, w = mask.shape
    colour_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, colour in LABEL_COLOURS.items():
        colour_mask[mask == label_id] = colour
    return Image.fromarray(colour_mask)

def overlay_mask_on_image(original_image, mask_image, alpha=0.4):
    original = original_image.convert("RGBA")
    mask = mask_image.convert("RGBA")
    
    blended = Image.blend(original, mask, alpha)
    return blended


def load_model(checkpoint_path, device, weights_only=True, num_classes=NUM_CLASSES):
    model = ImprovedUNet(n_classes=num_classes).to(device)
    if weights_only:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    return model


class PatchFolderDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.paths = [os.path.join(patch_dir, f"patch_{i}.png") for i in range(len(os.listdir(patch_dir))) 
              if os.path.exists(os.path.join(patch_dir, f"patch_{i}.png"))]

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8, 0.65, 0.8], std=[0.15, 0.15, 0.15])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def extract_patches_from_wsi(wsi_path, out_dir, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP,
    level=PATCH_LEVEL, tissue_threshold=TISSUE_THRESHOLD):

    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    from .utils import is_tissue_patch

    os.makedirs(out_dir, exist_ok=True)
    slide = TiffSlide(wsi_path)
    width, height = slide.level_dimensions[level]

    stride = int(round(patch_size * (1 - overlap)))
    slide_map = {}
    inference_flags = {}
    count = 0

    # debug overlay image
    scale_factor = 1 / 16  # use same factor in both directions
    thumb_width = int(width * scale_factor)
    thumb_height = int(height * scale_factor)
    downsampled = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
    draw = ImageDraw.Draw(downsampled)

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            # draw a box that is green for a tissue patch and red for non-tissue patch
            rect = [
                int(x * scale_factor),
                int(y * scale_factor),
                int((x + patch_size) * scale_factor),
                int((y + patch_size) * scale_factor),
            ]

            should_infer = is_tissue_patch(patch_np, tissue_threshold)
            patch.save(os.path.join(out_dir, f"patch_{count}.png"))

            slide_map[count] = (x, y, patch_size, patch_size)
            inference_flags[count] = should_infer

            draw.rectangle(rect, outline="green" if should_infer else "red", width=1)
            count += 1

    debug_path = os.path.join(out_dir, "patch_debug_overlay.png")
    downsampled.save(debug_path)

    patch_mask = np.zeros((height, width), dtype=np.uint8)
    for i, (x, y, w, h) in slide_map.items():
        if inference_flags.get(i, False):
            patch_mask[y:y+h, x:x+w] = 1
    Image.fromarray((patch_mask * 255).astype(np.uint8)).save(os.path.join(out_dir, "patch_inference_mask.png"))

    print(f"Total patches saved: {count}")
    print(f"Saved debug patch overlay: {debug_path}")
    return slide_map, inference_flags


def is_tissue_patch(patch_np, threshold=0.05):
    # HSV-based tissue filtering
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue_mask = (saturation > 20) & (value < 230)
    tissue_percentage = np.sum(tissue_mask) / (patch_np.shape[0] * patch_np.shape[1])
    return tissue_percentage > threshold

def load_all_patches_in_folder(patch_dir, transform=None):
    return PatchFolderDataset(patch_dir, transform)
