from uuid import uuid4
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from .improved_unet import ImprovedUNet
from .config import LABEL_COLOURS, PATCH_SIZE, PATCH_OVERLAP, TISSUE_THRESHOLD, PATCH_LEVEL, NUM_CLASSES
from typing import Tuple

# a hybrid approach, combining both connected component labelling and watershed
def get_instance_mask(
    mask: np.ndarray,
    tissue_type: int,
    *,
    min_size: int = 64,
    footprint_radius: int = 20,
    connectivity: int = 2,
    dtype_out: np.dtype = np.uint32,
) -> Tuple[np.ndarray, int]:

    # footprint_radius is a hyperparam - larger radius means fewer markers and less splitting
    # connectivity is used for the initial coarse labelling (connected component labelling)

    # quick exit
    binary = (mask == tissue_type)
    if not np.any(binary):
        return np.zeros_like(mask, dtype=dtype_out), 0

    # coarse connected components
    coarse_labels = measure.label(binary, connectivity=connectivity)

    # initialise, using uint32 gives 16 mill unique ids
    instance_mask = np.zeros(mask.shape, dtype=dtype_out)

    next_id: int = 1  # global label counter

    # iterate over each blob independently, more optimal for ram
    footprint = morphology.disk(footprint_radius)
    for region in measure.regionprops(coarse_labels):
        # remove specks to speed things up. min_size is smallest valid object size in pixels
        if region.area < min_size:
            continue
        # minimal bounding box slice
        slc_y, slc_x = region.slice
        blob_mask = binary[slc_y, slc_x]

        # distance map inside the blob
        dist = ndi.distance_transform_edt(blob_mask)

        # find seed points (local maxima of distance)
        # footprint defines minimum separation between peaks
        try:
            coords = peak_local_max(
                dist, footprint=footprint, labels=blob_mask, exclude_border=False
            )
            peaks = np.zeros_like(blob_mask, dtype=bool)
            if coords.size: # avoid empty peak array
                peaks[tuple(coords.T)] = True
        except TypeError:  # old api
            peaks = peak_local_max(
                dist, footprint=footprint, labels=blob_mask,
                indices=False, exclude_border=False
            )

        # fallback, if watershed would fail (rare, flat small blob) use CC
        markers, _ = ndi.label(peaks)
        if markers.max() == 0:                   # tiny flat blob fallback
            markers = measure.label(blob_mask)

        # marker‑controlled watershed to split touching objects
        labels_ws = segmentation.watershed(
            -dist,
            markers,
            mask=blob_mask,
            connectivity=connectivity
        )

        # write labels back into global array with offset
        if labels_ws.max() > 0:
            instance_mask_view = instance_mask[slc_y, slc_x]
            instance_ids = np.unique(labels_ws)
            instance_ids = instance_ids[instance_ids > 0]  # skip background

            for lab in instance_ids:
                instance_mask_view[labels_ws == lab] = next_id
                next_id += 1

        # cleanup
        del dist, peaks, markers, labels_ws

    num_instances = next_id - 1
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

