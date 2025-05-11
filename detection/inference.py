import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import ttach as tta
from tiffslide import TiffSlide

# Add parent directory to path for imports
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))

# Import InstanSeg modules
from instanseg.utils.utils import _move_channel_axis
from instanseg import InstanSeg
from instanseg.utils.pytorch_utils import torch_fastremap, centroids_from_lab, get_masked_patches
from instanseg.utils.pytorch_utils import _to_tensor_float32
from instanseg.inference_class import _rescale_to_pixel_size

# Constants
INSTANSEG_MODEL = "instanseg_brightfield_monkey.pt"
MODEL_NAMES = ["1952372.pt", "1950672.pt", "1949389_2.pt"]  # Public leaderboard #1 solution
DESTINATION_PIXEL_SIZE = 0.5
PATCH_SIZE = 128
USE_TTA = True
NORMALIZE_HE = False
RESCALE_OUTPUT = False if DESTINATION_PIXEL_SIZE == 0.5 else True
ORIGINAL_PIXEL_SIZE = 0.24199951445730394


class ModelEnsemble(torch.nn.Module):
    """Ensemble of multiple classification models with optional test-time augmentation."""
    
    def __init__(self, model_paths, device, use_tta=False):
        super().__init__()
        self.models = torch.nn.ModuleList([
            self.load_model(model_path, device, use_tta) for model_path in model_paths
        ])
        self.device = device
    
    def load_model(self, model_path, device, use_tta):
        model = torch.jit.load(model_path).eval().to(device)
        if use_tta:
            transforms = tta.Compose([
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),  
            ])
            model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
        return model
    
    def forward(self, x):
        with torch.no_grad():
            predictions = [model(x) for model in self.models]
            return torch.mean(torch.stack(predictions), dim=0)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on kidney transplant biopsy WSIs')
    parser.add_argument('--wsi_path', type=str, required=True, help='Path to the WSI file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model files')
    parser.add_argument('--bbox', type=int, nargs='+', 
                      help='Bounding box coordinates in format: ymin xmin ymax xmax [ymin xmin ymax xmax ...]')
    parser.add_argument('--bbox_file', type=str, help='Path to text file containing bounding box coordinates')
    return parser.parse_args()


def load_models(model_dir):
    """Load InstanSeg and classifier models."""
    # Load InstanSeg model
    instanseg_path = os.path.join(model_dir, INSTANSEG_MODEL)
    instanseg_script = torch.jit.load(instanseg_path).to("cuda")
    instanseg_model = InstanSeg(instanseg_script, verbosity=0)
    
    # Load classifier ensemble
    classifier_paths = [os.path.join(model_dir, name) for name in MODEL_NAMES]
    classifier = ModelEnsemble(
        model_paths=classifier_paths,
        device="cuda",
        use_tta=USE_TTA
    )
    
    return instanseg_model, classifier


def process_bbox(slide, bbox, instanseg_model, classifier):
    """Process a single bounding box region of the slide."""
    # Extract bbox coordinates
    y1, x1, y2, x2 = bbox
    bbox_native = [np.array([y1, x1]), np.array([y2, x2])]
    
    # Read region from slide
    image = slide.read_region((bbox_native[0][1], bbox_native[0][0]), 0, 
                             (bbox_native[1][1] - bbox_native[0][1], 
                              bbox_native[1][0] - bbox_native[0][0]), as_array=True)
    
    # Run InstanSeg model
    labels, input_tensor = instanseg_model.eval_medium_image(
        image, pixel_size=ORIGINAL_PIXEL_SIZE, 
        rescale_output=RESCALE_OUTPUT, 
        seed_threshold=0.1, 
        tile_size=1024
    )
    
    # Convert image to tensor
    tensor = _rescale_to_pixel_size(
        _to_tensor_float32(image), 
        ORIGINAL_PIXEL_SIZE, 
        DESTINATION_PIXEL_SIZE
    ).to("cpu")
    
    # Process labels and get masked patches
    labels = labels.to("cpu")
    labels = torch_fastremap(labels)
    
    # Check if labels contain any cells
    if labels.max() == 0:
        # Return empty results for bounding boxes with no cells
        return {
            'coords': np.array([]).reshape(0, 2),
            'classes': np.array([]),
            'confidences': np.array([]).reshape(0, 3)
        }
    
    crops, masks = get_masked_patches(labels, tensor.to("cpu"), patch_size=PATCH_SIZE)
    x = torch.cat((crops / 255.0, masks), dim=1)
    
    # Run classification
    batch_size = 1024
    y_hat = []
    
    with torch.amp.autocast("cuda"):
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_result = classifier(x[i:i+batch_size].float().to("cuda"))
                y_hat.append(batch_result)
    
    # Process classification results
    y_hat = torch.cat(y_hat, dim=0)
    y_hat = y_hat[:, -3:]  # Because of dual training
    y_hat = y_hat.cpu()
    
    # Get confidences and coordinates
    conf = y_hat.softmax(1)
    y_hat = y_hat.argmax(1)
    coords = centroids_from_lab(labels)[0]
    
    # Scale coordinates back to original scale
    coords_scaled = coords.cpu().numpy()[:, ::-1] * (DESTINATION_PIXEL_SIZE / ORIGINAL_PIXEL_SIZE) + bbox_native[0][::-1]
    
    return {
        'coords': coords_scaled,
        'classes': y_hat.cpu().numpy(),
        'confidences': conf.cpu().numpy()
    }


def create_output_dicts(all_coords, all_classes, all_confidences):
    """Create dictionaries for output JSON files."""
    # Initialize output dictionaries
    output_dicts = {
        "lymphocytes": {
            "name": "lymphocytes",
            "type": "Multiple points",
            "version": {"major": 1, "minor": 0},
            "points": [],
        },
        "monocytes": {
            "name": "monocytes",
            "type": "Multiple points",
            "version": {"major": 1, "minor": 0},
            "points": [],
        },
        "inflammatory-cells": {
            "name": "inflammatory-cells",
            "type": "Multiple points",
            "version": {"major": 1, "minor": 0},
            "points": [],
        }
    }
    
    # Fill dictionaries with detection points
    for idx, (coords, class_label, confidence) in enumerate(zip(all_coords, all_classes, all_confidences)):
        x, y = coords
        
        # Convert to millimeters
        x_mm = x * ORIGINAL_PIXEL_SIZE / 1000
        y_mm = y * ORIGINAL_PIXEL_SIZE / 1000
        
        # Create point records
        point_data = {
            "name": f"Point {idx}",
            "point": [x_mm, y_mm, ORIGINAL_PIXEL_SIZE],
        }
        
        # Add points with their probabilities
        inflammatory_point = point_data.copy()
        inflammatory_point["probability"] = float(confidence[0] + confidence[1])
        output_dicts["inflammatory-cells"]["points"].append(inflammatory_point)
        
        lymphocyte_point = point_data.copy()
        lymphocyte_point["probability"] = float(confidence[0])
        output_dicts["lymphocytes"]["points"].append(lymphocyte_point)
        
        monocyte_point = point_data.copy()
        monocyte_point["probability"] = float(confidence[1])
        output_dicts["monocytes"]["points"].append(monocyte_point)
    
    return (
        output_dicts["lymphocytes"],
        output_dicts["monocytes"],
        output_dicts["inflammatory-cells"]
    )


def save_outputs(output_dir, lymphocytes_dict, monocytes_dict, inflammatory_dict, image_path=None):
    """Save output JSON files and create record if needed."""
    # Save detection JSONs
    outputs = {
        "detected-lymphocytes.json": lymphocytes_dict,
        "detected-monocytes.json": monocytes_dict,
        "detected-inflammatory-cells.json": inflammatory_dict
    }
    
    for filename, content in outputs.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(content, f, indent=4)
        print(f"Saved {output_path}")
    
    # Create record JSON if image_path is provided (not for submission)
    if image_path:
        name = str(Path(image_path).name.replace("_PAS_CPG.tif", ""))
        record = {
            "pk": name,
            "inputs": [
                {
                    "image": {
                        "name": f"{name}_PAS_CPG.tif"
                    },
                    "interface": {
                        "slug": "kidney-transplant-biopsy",
                        "kind": "Image",
                        "super_kind": "Image",
                        "relative_path": "images/kidney-transplant-biopsy-wsi-pas"
                    }
                }
            ],
            "outputs": [
                {
                    "interface": {
                        "slug": output_slug,
                        "kind": "Multiple points",
                        "super_kind": "File",
                        "relative_path": output_path
                    }
                }
                for output_slug, output_path in {
                    "detected-lymphocytes": "detected-lymphocytes.json",
                    "detected-monocytes": "detected-monocytes.json",
                    "detected-inflammatory-cells": "detected-inflammatory-cells.json"
                }.items()
            ]
        }
        
        with open("./predictions.json", 'w') as f:
            json.dump([record], f, indent=4)
        print("Created predictions record")
        
        return [record]
    
    return []


def main():
    """Main function to run inference process."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing slide: {args.wsi_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Load models
    instanseg_model, classifier = load_models(args.model_dir)
    
    # Open slide
    slide = TiffSlide(args.wsi_path)
    
    # Process bounding boxes from file if provided
    bbox_list = []
    if args.bbox_file:
        if args.bbox:
            raise ValueError("Cannot provide both --bbox and --bbox_file")
        
        print(f"Reading bounding box coordinates from: {args.bbox_file}")
        with open(args.bbox_file, 'r') as f:
            for line in f:
                coords = list(map(int, line.strip().split()))
                if len(coords) != 4:
                    raise ValueError(f"Invalid bbox coordinates: {line.strip()}")
                bbox_list.extend(coords)
        args.bbox = bbox_list
    
    # Process bounding boxes
    all_results = []
    if args.bbox is not None:
        if len(args.bbox) % 4 != 0:
            raise ValueError(f"Number of bbox coordinates ({len(args.bbox)}) must be divisible by 4")
        
        # Process each bbox
        for i in range(0, len(args.bbox), 4):
            if i + 3 < len(args.bbox):
                bbox = args.bbox[i:i+4]
                print(f"Processing bbox: {bbox}")
                result = process_bbox(slide, bbox, instanseg_model, classifier)
                all_results.append(result)
    else:
        # Process entire slide (not implemented in original code)
        print("No bounding boxes provided. Please specify bounding boxes with --bbox or --bbox_file.")
        return 1
    
    # Combine results from all bounding boxes
    all_coords = np.concatenate([r['coords'] for r in all_results]) if all_results else np.array([])
    all_classes = np.concatenate([r['classes'] for r in all_results]) if all_results else np.array([])
    all_confidences = np.concatenate([r['confidences'] for r in all_results]) if all_results else np.array([])
    
    # Create and save output files
    lymphocytes_dict, monocytes_dict, inflammatory_dict = create_output_dicts(
        all_coords, all_classes, all_confidences)
    
    save_outputs(args.output_dir, lymphocytes_dict, monocytes_dict, inflammatory_dict, args.wsi_path)
    
    print(f"Inference completed. Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())