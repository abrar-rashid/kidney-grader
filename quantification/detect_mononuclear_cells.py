import numpy as np
from segmentation.config import PATCH_SIZE
import torch
import cv2
from pathlib import Path
import logging
from PIL import Image
from ultralytics import YOLO

class MononuclearCellDetector:
    # currently YOLOv8 model (to be replaced by instanseg later) to detect mononuclear inflammatory cells in tubules
    
    def __init__(self, model_path="checkpoints/yolov8_mononuclear.pt", 
                 conf_threshold=0.25, iou_threshold=0.45):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold # confidence threshold for detections
        self.model.iou = iou_threshold # IoU threshold for non-maximum suppression (NMS)
        
    def preprocess_image(self, image, patch_size=PATCH_SIZE): # to preprocess image into list of patches and their positions
        if len(image.shape) == 2: # if image is greyscale then need to convert to RGB for YOLO
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]
        patches = []
        positions = []
        
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = image[y:min(y + patch_size, h), 
                             x:min(x + patch_size, w)]
                #Zero padding when needed in order to pass into YOlO
                if patch.shape[:2] != (patch_size, patch_size):
                    pad_h = patch_size - patch.shape[0]
                    pad_w = patch_size - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)))
                
                patches.append(patch)
                positions.append((y, x))
        
        return patches, positions
    
    def postprocess_detections(self, detections, positions, original_shape):
        # takes YOLO detection results, patch positions list and original shape to give 
        # array of coords for detected nuclei relative to whole WSI
        nuclei_coords = []
        
        for det, (y, x) in zip(detections, positions): #for each patch's detection, convert
            # bbox centres to coords and adds patch offset
            if len(det.boxes) > 0:
                boxes = det.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    center_x += x
                    center_y += y
                    
                    # original image bounds check
                    if (0 <= center_x < original_shape[1] and 
                        0 <= center_y < original_shape[0]):
                        nuclei_coords.append([center_y, center_x])
        
        return np.array(nuclei_coords)
    
    def detect(self, image, batch_size=4):
        patches, positions = self.preprocess_image(image)
        
        all_detections = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            with torch.no_grad():
                detections = self.model(batch, verbose=False)
                all_detections.extend(detections)
        
        nuclei_coords = self.postprocess_detections(all_detections, positions, image.shape)
        nuclei_coords = self.apply_NMS(nuclei_coords, iou_threshold=0.3)
        
        logging.info(f"Detected {len(nuclei_coords)} mononuclear cells")
        return nuclei_coords
    
    def apply_NMS(self, coords, iou_threshold=0.3): 
        # nms removes duplicate detections

        if len(coords) == 0:
            return coords
        boxes = np.zeros((len(coords), 4))
        boxes[:, 0] = coords[:, 1] - 5
        boxes[:, 1] = coords[:, 0] - 5
        boxes[:, 2] = 10
        boxes[:, 3] = 10
        
        indices = cv2.dnn.NMSBoxes(boxes, np.ones(len(coords)), 0.5, iou_threshold)
        
        return coords[indices]

def detect_mononuclear_cells(wsi_path, model_path="checkpoints/yolov8_mononuclear.pt", device=None, batch_size=4):
    detector = MononuclearCellDetector(model_path=model_path)
    
    image = np.array(Image.open(wsi_path).convert("RGB"))
    nuclei_coords = detector.detect(image, batch_size=batch_size)
    
    vis_image = image.copy()
    for y, x in nuclei_coords:
        cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)

    vis_path = Path(wsi_path).with_name("detection_vis.png")
    Image.fromarray(vis_image).save(vis_path)

    return {
        "coordinates": nuclei_coords,
        "visualization_path": str(vis_path)
    }

# def run_detect(in_path, out_path): # run detection on full WSI
#     image = np.array(Image.open(in_path).convert("RGB"))
    
#     nuclei_coords = detect_mononuclear_cells(image)
    
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.save(out_path / "nuclei_coords.npy", nuclei_coords)
    
#     vis_image = image.copy()
#     for y, x in nuclei_coords:
#         cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
#     Image.fromarray(vis_image).save(out_path / "detection_vis.png")
    
#     logging.info(f"Saved detection results to {out_path}")
