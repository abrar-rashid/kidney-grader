# Credit to Claude LLM for patch extraction and feature extraction code

"""
Enhanced patch extractor for kidney biopsy WSIs
Optimized for single-stage tubulitis scoring pipeline
"""

import os
import cv2
import json
import random
import numpy as np
import pandas as pd
import openslide
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import h5py

# Tissue detection parameters
MASK_SAT = 0
MASK_VAL = 245

class KidneyPatchExtractor:
    """Enhanced patch extractor for kidney biopsy WSIs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patch_size = config['patch_extraction']['patch_size']
        self.level = config['patch_extraction']['magnification_level']
        self.overlap = config['patch_extraction']['overlap']
        self.tissue_threshold = config['patch_extraction']['tissue_threshold']
        self.max_patches = config['patch_extraction']['max_patches_per_wsi']
        self.random_seed = config['sampling']['random_seed']
        
        # Quality control parameters
        self.min_tissue_area = config['quality_control']['min_tissue_area']
        self.blur_threshold = config['quality_control']['blur_threshold']
        self.saturation_threshold = config['quality_control']['saturation_threshold']
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    def is_tissue_patch(self, patch_np: np.ndarray, threshold: float = 0.15) -> bool:
        """Determine if patch contains sufficient tissue - optimized for PAS-stained kidney"""
        # Convert to different color spaces for better tissue detection
        hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(patch_np, cv2.COLOR_RGB2LAB)
        
        # HSV-based tissue filtering (original method)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hsv_tissue_mask = (saturation > MASK_SAT) & (value < MASK_VAL)
        
        # LAB-based tissue detection (better for PAS stains)
        # PAS shows pink/purple tissue, avoid white/very pale areas
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        
        # Tissue is typically darker (lower L) and has some color (non-zero A, B)
        lab_tissue_mask = (l_channel < 200) & (np.abs(a_channel - 128) > 5)
        
        # Combine both methods
        combined_mask = hsv_tissue_mask | lab_tissue_mask
        tissue_percentage = np.sum(combined_mask) / (patch_np.shape[0] * patch_np.shape[1])
        
        return tissue_percentage > threshold
    
    def is_quality_patch(self, patch_np: np.ndarray) -> bool:
        """Check patch quality (blur, artifacts, etc.)"""
        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.blur_threshold:
            return False
            
        # Check saturation (avoid very pale/washed out patches)
        hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        
        return mean_saturation > self.saturation_threshold
    
    def extract_tissue_mask(self, slide: openslide.OpenSlide) -> np.ndarray:
        """Extract tissue mask from WSI thumbnail"""
        # Get thumbnail for tissue detection
        level_dims = slide.level_dimensions
        width, height = level_dims[self.level]
        
        scale_factor = 1 / min(32, width // 4000) if width > 4000 else 1 / 16
        thumb_width = int(width * scale_factor)
        thumb_height = int(height * scale_factor)
        
        thumbnail = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
        thumbnail_np = np.array(thumbnail)
        
        # Apply tissue detection
        hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        tissue_mask = (saturation > MASK_SAT) & (value < MASK_VAL)
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        
        return tissue_mask, scale_factor, thumbnail_np
    
    def extract_patches_from_wsi(self, wsi_path: str) -> Dict[str, Any]:
        """Extract patches from WSI using Vishal's approach"""
        slide_name = Path(wsi_path).stem
        print(f"Processing {slide_name}...")
        
        # Get sampling strategy and extraction mode from config
        sampling_strategy = self.config['sampling'].get('strategy', 'contiguous')
        extraction_mode = self.config['sampling'].get('extraction_mode', 'contiguous')
        
        # For now, implement whole tissue sampling with different extraction modes
        # TODO: Implement tubular_focused when segmentation masks are available
        if sampling_strategy == "tubular_focused":
            print(f"  Warning: Tubular focused sampling not yet implemented, falling back to whole tissue")
            tubular_config = self.config['sampling']['tubular_focused']
            extraction_mode = tubular_config.get('extraction_mode', 'random')
        
        try:
            # Suppress TIFF warnings (they're not fatal)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                slide = openslide.OpenSlide(wsi_path)
            
            width, height = slide.level_dimensions[self.level]
            
            # Calculate downsample factor based on slide size
            scale_factor = 1 / min(32, width // 4000) if width > 4000 else 1 / 16
            thumb_width = int(width * scale_factor)
            thumb_height = int(height * scale_factor)

            print(f"  Creating thumbnail at resolution {thumb_width}x{thumb_height}")
            thumbnail = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
            thumbnail_np = np.array(thumbnail)

            # Apply tissue detection to thumbnail
            hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            tissue_mask = (saturation > MASK_SAT) & (value < MASK_VAL)

            # Clean up the mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

            # Calculate thumbnail patch size
            thumb_patch_size = int(self.patch_size * scale_factor)
            
            patches = []
            coordinates = []
            attempts = 0

            if extraction_mode == "contiguous":
                print(f"  Extracting contiguous patches from tissue regions")
                
                # Calculate step size (with overlap)
                step_size = int(self.patch_size * (1 - self.overlap))
                
                # Calculate number of patches in each dimension
                num_patches_x = (width - self.patch_size) // step_size + 1
                num_patches_y = (height - self.patch_size) // step_size + 1
                
                print(f"  Creating {num_patches_x}x{num_patches_y} grid with step size {step_size}")
                
                # Create a dilated tissue mask to capture regions near tissue
                dilated_tissue_mask = tissue_mask.copy()
                dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated_tissue_mask = cv2.dilate(dilated_tissue_mask.astype(np.uint8), dilation_kernel, iterations=1)
                
                # Create grid positions filtered by tissue
                roi_positions = []
                for y_idx in range(num_patches_y):
                    for x_idx in range(num_patches_x):
                        # Calculate thumbnail coordinates
                        thumb_x = int(x_idx * step_size * scale_factor)
                        thumb_y = int(y_idx * step_size * scale_factor)
                        
                        # Skip if outside thumbnail bounds
                        if (thumb_x + thumb_patch_size >= thumbnail_np.shape[1] or 
                            thumb_y + thumb_patch_size >= thumbnail_np.shape[0]):
                            continue
                            
                        # Check if this patch overlaps with tissue
                        patch_mask = dilated_tissue_mask[thumb_y:thumb_y + thumb_patch_size,
                                                        thumb_x:thumb_x + thumb_patch_size]
                        if np.sum(patch_mask) > 0:
                            roi_positions.append((x_idx, y_idx, thumb_x, thumb_y))
                
                print(f"  Found {len(roi_positions)} potential tissue positions")
                
                # Process ROI positions
                count = 0
                for x_idx, y_idx, thumb_x, thumb_y in roi_positions:
                    if count >= self.max_patches:
                        break
                        
                    attempts += 1
                    
                    # Calculate full resolution coordinates
                    full_x = x_idx * step_size
                    full_y = y_idx * step_size
                    
                    # Check tissue percentage in thumbnail
                    region = tissue_mask[thumb_y:thumb_y + thumb_patch_size,
                                       thumb_x:thumb_x + thumb_patch_size]
                    tissue_percentage = np.sum(region) / region.size
                    
                    if tissue_percentage <= self.tissue_threshold:
                        continue
                    
                    # Extract full resolution patch
                    try:
                        patch_pil = slide.read_region((full_x, full_y), self.level,
                                                    (self.patch_size, self.patch_size)).convert("RGB")
                        patch_np = np.array(patch_pil)
                        
                        # Final verification
                        if not self.is_tissue_patch(patch_np, threshold=0.15):
                            continue
                        
                        patches.append(patch_np)
                        coordinates.append((full_x, full_y))
                        count += 1
                        
                    except Exception as e:
                        continue
                        
            else:  # random mode
                # Find coordinates of all tissue pixels in the thumbnail
                tissue_coords = np.where(tissue_mask)
                tissue_points = list(zip(tissue_coords[1], tissue_coords[0]))  # (x, y) format

                if not tissue_points:
                    print(f"  No tissue regions found in {slide_name}")
                    slide.close()
                    return {"patches": [], "coordinates": [], "slide_name": slide_name, "status": "no_tissue"}

                print(f"  Randomly sampling up to {self.max_patches} patches from tissue regions")

                # Keep track of already sampled regions to avoid overlap
                sampled_regions = set()
                max_attempts = self.max_patches * 100  # Limit attempts to avoid infinite loop
                count = 0

                while count < self.max_patches and attempts < max_attempts:
                    attempts += 1

                    # Randomly select a tissue point from the mask
                    if not tissue_points:
                        break

                    point_idx = np.random.randint(0, len(tissue_points))
                    thumb_x, thumb_y = tissue_points[point_idx]

                    # Check if we have enough space for a patch
                    if (thumb_x + thumb_patch_size >= thumbnail_np.shape[1] or 
                        thumb_y + thumb_patch_size >= thumbnail_np.shape[0]):
                        continue

                    # Verify this region has enough tissue
                    region = tissue_mask[thumb_y:thumb_y + thumb_patch_size,
                                       thumb_x:thumb_x + thumb_patch_size]
                    tissue_percentage = np.sum(region) / region.size

                    if tissue_percentage <= self.tissue_threshold:
                        continue

                    # Map to full resolution coordinates
                    full_x = int(thumb_x / scale_factor)
                    full_y = int(thumb_y / scale_factor)

                    # Create a region key to avoid overlap
                    region_key = (full_x // (self.patch_size // 4), full_y // (self.patch_size // 4))
                    if region_key in sampled_regions:
                        continue

                    sampled_regions.add(region_key)

                    # Extract full resolution patch
                    try:
                        patch_pil = slide.read_region((full_x, full_y), self.level, 
                                                    (self.patch_size, self.patch_size)).convert("RGB")
                        patch_np = np.array(patch_pil)

                        # Final verification on the full resolution patch
                        if not self.is_tissue_patch(patch_np, threshold=0.15):
                            continue

                        # Store patch with coordinates
                        patches.append(patch_np)
                        coordinates.append((full_x, full_y))
                        count += 1

                    except Exception as e:
                        # Don't print every extraction error to reduce noise
                        continue

            slide.close()

            # Calculate total tissue area
            total_tissue_pixels = np.sum(tissue_mask) / (scale_factor ** 2)
            
            # Add diagnostic info for low patch counts
            if len(patches) < 100 and total_tissue_pixels > 100000000:
                print(f"  WARNING: Only {len(patches)} patches from {total_tissue_pixels:.0f} tissue pixels")
                print(f"  Tried {attempts} attempts")

            print(f"  Extracted {len(patches)} patches from {slide_name} after {attempts} attempts")
            
            return {
                "patches": patches,
                "coordinates": coordinates,
                "slide_name": slide_name,
                "status": "success",
                "tissue_area": total_tissue_pixels,
                "thumbnail": thumbnail_np
            }
            
        except Exception as e:
            print(f"  Error processing {slide_name}: {e}")
            return {"patches": [], "coordinates": [], "slide_name": slide_name, "status": f"error: {e}"}
    
    def save_patch_data(self, patch_data: Dict[str, Any], output_dir: Path) -> str:
        """Save patch data and metadata"""
        slide_name = patch_data["slide_name"]
        patches = patch_data["patches"]
        coordinates = patch_data["coordinates"]
        
        # Create slide-specific directory
        slide_dir = output_dir / slide_name
        slide_dir.mkdir(parents=True, exist_ok=True)
        
        # Save patches as HDF5 file for efficient storage
        h5_path = slide_dir / f"{slide_name}_patches.h5"
        
        with h5py.File(h5_path, 'w') as f:
            # Save patches
            if patches:
                patches_array = np.stack(patches)
                f.create_dataset('patches', data=patches_array, compression='gzip')
                
                # Save coordinates
                coords_array = np.array(coordinates)
                f.create_dataset('coordinates', data=coords_array)
                
                # Save metadata
                f.attrs['slide_name'] = slide_name
                f.attrs['num_patches'] = len(patches)
                f.attrs['patch_size'] = self.patch_size
                f.attrs['level'] = self.level
                f.attrs['status'] = patch_data["status"]
                if 'tissue_area' in patch_data:
                    f.attrs['tissue_area'] = patch_data["tissue_area"]
        
        # Save thumbnail for visualization
        if 'thumbnail' in patch_data:
            thumbnail_path = slide_dir / f"{slide_name}_thumbnail.png"
            Image.fromarray(patch_data['thumbnail']).save(thumbnail_path)
        
        # Save coordinate JSON for easy access
        coord_json_path = slide_dir / f"{slide_name}_coordinates.json"
        with open(coord_json_path, 'w') as f:
            json.dump({
                "slide_name": slide_name,
                "coordinates": coordinates,
                "num_patches": len(patches),
                "status": patch_data["status"]
            }, f, indent=2)
        
        return str(h5_path)
    
    def process_wsi_list(self, wsi_paths: List[str], output_dir: str) -> Dict[str, Any]:
        """Process multiple WSIs and extract patches"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "processed_slides": [],
            "failed_slides": [],
            "total_patches": 0,
            "summary": {}
        }
        
        for wsi_path in tqdm(wsi_paths, desc="Extracting patches"):
            patch_data = self.extract_patches_from_wsi(wsi_path)
            
            if patch_data["status"] == "success" and len(patch_data["patches"]) > 0:
                h5_path = self.save_patch_data(patch_data, output_path)
                results["processed_slides"].append({
                    "slide_name": patch_data["slide_name"],
                    "num_patches": len(patch_data["patches"]),
                    "h5_path": h5_path,
                    "tissue_area": patch_data.get("tissue_area", 0)
                })
                results["total_patches"] += len(patch_data["patches"])
            else:
                results["failed_slides"].append({
                    "slide_name": patch_data["slide_name"],
                    "status": patch_data["status"]
                })
        
        # Save summary
        results["summary"] = {
            "total_slides_processed": len(wsi_paths),
            "successful_slides": len(results["processed_slides"]),
            "failed_slides": len(results["failed_slides"]),
            "total_patches_extracted": results["total_patches"],
            "average_patches_per_slide": results["total_patches"] / max(1, len(results["processed_slides"]))
        }
        
        # Save results summary
        summary_path = output_path / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results 