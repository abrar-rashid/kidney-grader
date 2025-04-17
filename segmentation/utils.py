import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from tiffslide import TiffSlide
from torchvision import transforms
import torch
import logging
from scipy.ndimage import label
from .improved_unet import ImprovedUNet
from .config import LABEL_COLOURS, PATCH_SIZE, PATCH_OVERLAP, TISSUE_THRESHOLD, PATCH_LEVEL, NUM_CLASSES


def get_instance_mask(mask, tissue_type=1):
    # get instance segmentation mask for a specific tissue type using watershed algorithm.
    binary = get_binary_class_mask(mask, tissue_type)  # 0/1 mask for selected class

    # clean up with morphological opening (removes noise, but doesn't merge objects)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # distance transform for separating close regions
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 1, 0)
    sure_fg = np.uint8(sure_fg)

    # sure background via dilation
    sure_bg = cv2.dilate(opened, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # ensure background is 1 instead of 0
    markers[unknown == 1] = 0

    # watershed needs a 3-channel image
    color_img = cv2.cvtColor((opened * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    # convert markers to instance mask: 0 = background, 1...n = instance ids
    instance_mask = np.where(markers > 1, markers - 1, 0).astype(np.uint16)
    num_instances = instance_mask.max()

    logging.info(f"Found {num_instances} instances of tissue type {tissue_type}")
    return instance_mask, num_instances


def get_binary_class_mask(mask, class_idx):
    # convert multi-class mask to binary mask for a specific class.
    return (mask == class_idx).astype(np.uint8)


def create_visualization(mask, original_image=None, alpha=0.4):
    #Create color-coded mask and optionally overlay it on the original image
    h, w = mask.shape
    colour_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, colour in LABEL_COLOURS.items():
        colour_mask[mask == label_id] = colour
    colour_mask = Image.fromarray(colour_mask)
    
    if original_image is not None:
        original = original_image.convert("RGBA")
        mask_rgba = colour_mask.convert("RGBA")
        return Image.blend(original, mask_rgba, alpha)
    return colour_mask


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
    #dataset class for loading patches from a folder
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
    level=PATCH_LEVEL, tissue_threshold=TISSUE_THRESHOLD, tissue_filtering=False):
    # Extracts patches from a Whole Slide Image and save them to disk
    os.makedirs(out_dir, exist_ok=True)
    slide = TiffSlide(wsi_path)
    width, height = slide.level_dimensions[level]

    stride = int(round(patch_size * (1 - overlap)))
    slide_map = {}
    inference_flags = {}
    count = 0

    from tqdm import tqdm

    coords = [
        (x, y)
        for y in range(0, height - patch_size + 1, stride)
        for x in range(0, width - patch_size + 1, stride)
    ]

    for x, y in tqdm(coords, desc="Extracting Patches", unit="patch"):
        patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
        patch_np = np.array(patch)

        should_infer = is_tissue_patch(patch_np, tissue_threshold) if tissue_filtering else True
        patch.save(os.path.join(out_dir, f"patch_{count}.png"))

        slide_map[count] = (x, y, patch_size, patch_size)
        inference_flags[count] = should_infer

        count += 1

    patch_mask = np.zeros((height, width), dtype=np.uint8)
    for i, (x, y, w, h) in slide_map.items():
        if inference_flags.get(i, False):
            patch_mask[y:y+h, x:x+w] = 1
    Image.fromarray((patch_mask * 255).astype(np.uint8)).save(os.path.join(out_dir, "patch_inference_mask.png"))

    print(f"Total patches saved: {count}")
    return slide_map, inference_flags


def is_tissue_patch(patch_np, threshold=0.05):
    # TODO FIX
    # determine if a patch contains enough tissue using HSV color space analysis
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue_mask = (saturation > 20) & (value < 230)
    tissue_percentage = np.sum(tissue_mask) / (patch_np.shape[0] * patch_np.shape[1])
    return tissue_percentage > threshold

def load_all_patches_in_folder(patch_dir, transform=None):
    return PatchFolderDataset(patch_dir, transform)
