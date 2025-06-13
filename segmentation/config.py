DEFAULT_SEGMENTATION_MODEL_PATH = "checkpoints/segmentation/kidney_grader_unet.pth"
DEFAULT_YOLO_MODEL_PATH = "checkpoints/yolov8_mononuclear.pt"

PATCH_SIZE = 512
PATCH_OVERLAP = 0.25
TISSUE_THRESHOLD = 0.05
PATCH_LEVEL = 0

SEGMENTATION_OUTPUT_DIR = "segmentation/outputs"
QUANTIFICATION_OUTPUT_DIR = "quantification/outputs"
GRADING_OUTPUT_DIR = "grading/outputs"

NUM_CLASSES = 5
LABEL_COLOURS = {
    0: (0, 0, 0),       # background
    1: (255, 0, 0),     # tubuli
    2: (0, 255, 0),     # vein
    3: (0, 0, 255),     # artery
    4: (0, 255, 255),   # glomeruli
}

CLASS_NAMES = ['Background', 'Tubuli', 'Vein', 'Artery', 'Glomeruli']
