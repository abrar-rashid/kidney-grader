import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import time
import cv2
from rich.console import Console
from rich.progress import Progress
from grading.banff_grade import calculate_tubulitis_score
from grading.quantify import analyze_tubule_cell_distribution, convert_numpy_types, count_cells_in_tubules

console = Console()

def setup_logging(output_dir: Path) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pipeline.log"),
            logging.StreamHandler()
        ]
    )

class KidneyGraderPipeline:
    # Main pipeline for kidney biopsy grading
    
    def __init__(self, output_dir: str, model_path: str = "checkpoints/segmentation/kidney_grader_unet.pth", prob_thres: float = 0.50, custom_detection_json: str = None, custom_instance_mask_class1: str = None):
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.prob_thres = prob_thres
        self.custom_detection_json = custom_detection_json
        self.custom_instance_mask_class1 = custom_instance_mask_class1

        setup_logging(self.output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create main directory structure
        self.individual_reports_dir = self.output_dir / "individual_reports"
        self.summary_dir = self.output_dir / "summary"
        
        # Create main directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.individual_reports_dir.mkdir(exist_ok=True)
        self.summary_dir.mkdir(exist_ok=True)

    def validate_wsi_file(self, wsi_path: str) -> Dict[str, Any]:
        """Validate WSI file before processing to detect corruption or other issues.
        
        Returns:
            Dict with validation results: {'valid': bool, 'error': str, 'size_mb': float, 'dimensions': tuple}
        """
        wsi_path = Path(wsi_path)
        
        # Check if file exists
        if not wsi_path.exists():
            return {'valid': False, 'error': f'File does not exist: {wsi_path}', 'size_mb': 0, 'dimensions': None}
        
        # Check file size
        try:
            file_size_mb = wsi_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            return {'valid': False, 'error': f'Cannot read file size: {e}', 'size_mb': 0, 'dimensions': None}
        
        # Check if file is too small (likely corrupted) or too large (might cause issues)
        if file_size_mb < 10:  # Less than 10MB is suspicious for a WSI
            return {'valid': False, 'error': f'File too small ({file_size_mb:.1f} MB), likely corrupted', 'size_mb': file_size_mb, 'dimensions': None}
        
        if file_size_mb > 50000:  # Greater than 50GB is extremely large
            return {'valid': False, 'error': f'File extremely large ({file_size_mb:.1f} MB), processing may fail', 'size_mb': file_size_mb, 'dimensions': None}
        
        # Try to open the file with TiffSlide to check for TIFF corruption
        try:
            from tiffslide import TiffSlide
            slide = TiffSlide(str(wsi_path))
            
            # Check if we can read basic properties
            dimensions = slide.dimensions
            level_count = slide.level_count
            
            # Check for reasonable dimensions
            if dimensions[0] <= 0 or dimensions[1] <= 0:
                slide.close()
                return {'valid': False, 'error': f'Invalid dimensions: {dimensions}', 'size_mb': file_size_mb, 'dimensions': dimensions}
            
            # Check if dimensions are extremely large (might cause memory issues)
            total_pixels = dimensions[0] * dimensions[1]
            if total_pixels > 500_000_000_000:  # 500 billion pixels
                slide.close()
                return {'valid': False, 'error': f'Extremely large image ({total_pixels:,} pixels), may cause memory issues', 'size_mb': file_size_mb, 'dimensions': dimensions}
            
            # Try to read a small region to test for TIFF directory corruption
            try:
                test_region = slide.read_region((0, 0), slide.level_count - 1, (100, 100))
                test_region.close()
            except Exception as e:
                slide.close()
                return {'valid': False, 'error': f'TIFF directory corruption detected: {e}', 'size_mb': file_size_mb, 'dimensions': dimensions}
            
            slide.close()
            
            self.logger.info(f"WSI validation passed: {wsi_path.name} ({file_size_mb:.1f} MB, {dimensions[0]}x{dimensions[1]})")
            return {'valid': True, 'error': None, 'size_mb': file_size_mb, 'dimensions': dimensions}
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'tiff' in error_msg and ('directory' in error_msg or 'allocate memory' in error_msg):
                return {'valid': False, 'error': f'TIFF file corrupted: {e}', 'size_mb': file_size_mb, 'dimensions': None}
            else:
                return {'valid': False, 'error': f'Cannot open WSI file: {e}', 'size_mb': file_size_mb, 'dimensions': None}

    def get_output_paths(self, wsi_path: str) -> Dict[str, Path]:
        wsi_name = Path(wsi_path).stem
        
        # Create parameter-specific tag for directories
        prob_tag = f"p{self.prob_thres:.2f}".replace(".", "")
        param_tag = f"{prob_tag}"
        
        # Create WSI-specific directory
        wsi_dir = self.individual_reports_dir / wsi_name
        wsi_dir.mkdir(exist_ok=True)
        
        # Create shared directories for segmentation and detection (param-independent)
        segmentation_dir = wsi_dir / "segmentation"
        detection_dir = wsi_dir / "detection"
        segmentation_dir.mkdir(exist_ok=True)
        detection_dir.mkdir(exist_ok=True)
        
        # Create parameter-specific grading directory
        grading_dir = wsi_dir / "grading" / param_tag
        grading_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "wsi_name": wsi_name,
            "param_tag": param_tag,
            "tubule_mask": segmentation_dir / f"{wsi_name}_full_instance_mask_class1.tiff",
            "inflam_cell_mask": detection_dir / "detected-inflammatory-cells.json",
            "counts_csv": grading_dir / "tubule_counts.csv",
            "quant_json": grading_dir / "quantification.json",
            "grading_report": grading_dir / "grading_report.json"
        }
    
    def run_stage1(self, wsi_path: str, force: bool = False, visualise: bool = False) -> Dict[str, Any]:
        from segmentation.segment import run_segment

        self.logger.info(f"Running Stage 1: Segmentation for {wsi_path}")
        wsi_name = Path(wsi_path).stem
        
        # Validate WSI file before processing
        validation = self.validate_wsi_file(wsi_path)
        if not validation['valid']:
            error_msg = f"WSI validation failed for {wsi_name}: {validation['error']}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Log validation info
        if validation['dimensions']:
            self.logger.info(f"WSI validated: {validation['size_mb']:.1f} MB, {validation['dimensions'][0]}x{validation['dimensions'][1]} pixels")
        
        paths = self.get_output_paths(wsi_path)
        
        # Use the shared segmentation directory
        output_dir = self.individual_reports_dir / wsi_name / "segmentation"
        semantic_mask_path = output_dir / f"{wsi_name}_semantic_mask.tiff"
        instance_mask_path1 = output_dir / f"{wsi_name}_full_instance_mask_class1.tiff"

        # Check if custom instance masks are provided
        custom_masks_used = False
        if self.custom_instance_mask_class1:
            output_dir.mkdir(exist_ok=True)
            
            # Handle custom class 1 instance mask (tubules)
            custom_mask_path1 = Path(self.custom_instance_mask_class1)
            
            # Check if both files exist
            if custom_mask_path1.exists():
                self.logger.info(f"Using custom instance mask:")
                self.logger.info(f"  - Class 1 (tubules): {custom_mask_path1}")
                
                # Copy both files
                import shutil
                shutil.copy(custom_mask_path1, instance_mask_path1)
                custom_masks_used = True
                
                # Return early with custom masks
                instance_mask_paths = {
                    1: str(instance_mask_path1)
                }
                return {
                    "semantic_mask_path": str(semantic_mask_path) if semantic_mask_path.exists() else None,
                    "instance_mask_paths": instance_mask_paths,
                    "custom_masks": True
                }
            else:
                missing_files = []
                if not custom_mask_path1.exists():
                    missing_files.append(f"Class 1 mask: {custom_mask_path1}")
                    
                self.logger.warning(f"Custom instance mask not found: {', '.join(missing_files)}. Will run segmentation.")

        # check if both instance masks exist before skipping
        if instance_mask_path1.exists() and not force:
            self.logger.info(f"Segmentation mask already exists at {output_dir}, skipping segmentation.")
            instance_mask_paths = {
                int(p.stem.split("class")[-1]): str(p)
                for p in output_dir.glob(f"{wsi_name}_full_instance_mask_class*.tiff")
            }
            return {
                "semantic_mask_path": str(semantic_mask_path),
                "instance_mask_paths": instance_mask_paths
            }

        # run segmentation if not cached - wrap in try-except for better error handling
        try:
            self.logger.info(f"Starting segmentation for {wsi_name} (file size: {validation['size_mb']:.1f} MB)")
            result = run_segment(wsi_path, output_dir=output_dir, model_path=self.model_path, visualise=visualise)
            self.logger.info(f"Segmentation completed successfully for {wsi_name}")
            return result
        except Exception as e:
            error_msg = f"Segmentation failed for {wsi_name}: {e}"
            self.logger.error(error_msg)
            # Clean up any partial files that might have been created
            try:
                if output_dir.exists():
                    for file in output_dir.glob("*"):
                        if file.is_file():
                            file.unlink()
                            self.logger.info(f"Cleaned up partial file: {file}")
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to clean up partial files: {cleanup_error}")
            raise RuntimeError(error_msg)

    def run_stage2(self, wsi_path, force: bool = False, visualise: bool = False) -> dict:
        from detection.detect import run_inflammatory_cell_detection

        self.logger.info("Running Stage 2: Inflammatory cell detection")
        wsi_name = Path(wsi_path).stem
        paths = self.get_output_paths(wsi_path)
        
        # Use the shared detection directory
        output_dir = self.individual_reports_dir / wsi_name / "detection"
        
        json_detection_path = output_dir / "detected-inflammatory-cells.json"

        # If custom detection JSON is provided, use it instead
        if self.custom_detection_json:
            custom_json_path = Path(self.custom_detection_json)
            if custom_json_path.exists():
                self.logger.info(f"Using custom detection JSON from {custom_json_path}")
                output_dir.mkdir(exist_ok=True)
                
                # Copy the custom JSON to the expected location
                import shutil
                shutil.copy(custom_json_path, json_detection_path)
                
                with open(json_detection_path) as f:
                    inflam = json.load(f)

                # Load all points first
                all_points = inflam["points"]
                total_points = len(all_points)
                
                # Filter points based on probability threshold
                filtered_points = [
                    pt for pt in all_points 
                    if "probability" in pt and pt["probability"] >= self.prob_thres
                ]
                
                # Extract coordinates from filtered points
                mm_coords = np.array(
                    [[pt["point"][0], pt["point"][1]] for pt in filtered_points],
                    dtype=np.float32,
                )
                
                self.logger.info(f"Filtered inflammatory cells from {total_points} to {len(filtered_points)} using threshold {self.prob_thres}")
                
                # Generate visualization if requested
                if visualise:
                    self._run_detection_visualization(wsi_path, mm_coords, output_dir)
                
                return {
                    "wsi_name": wsi_name,
                    "prob_threshold": self.prob_thres,
                    "inflam_cell_coords_path": str(json_detection_path),
                    "custom_detection": True
                }
            else:
                self.logger.warning(f"Custom detection JSON {custom_json_path} not found. Falling back to standard detection.")

        # check if file exists and log the details
        self.logger.info(f"Checking if detection file exists at: {json_detection_path}")
        if json_detection_path.exists() and not force:
            self.logger.info(f"Inflammatory cell detection already exists at {json_detection_path}, skipping Stage 2.")
            with open(json_detection_path) as f:
                inflam = json.load(f)

            # Load all points first
            all_points = inflam["points"]
            total_points = len(all_points)
            
            # Filter points based on probability threshold
            filtered_points = [
                pt for pt in all_points 
                if "probability" in pt and pt["probability"] >= self.prob_thres
            ]
            
            # Extract coordinates from filtered points
            mm_coords = np.array(
                [[pt["point"][0], pt["point"][1]] for pt in filtered_points],
                dtype=np.float32,
            )
            
            self.logger.info(f"Filtered inflammatory cells from {total_points} to {len(filtered_points)} using threshold {self.prob_thres}")
            
            # Generate visualization if requested
            if visualise:
                self._run_detection_visualization(wsi_path, mm_coords, output_dir)
            
            return {
                "wsi_name": wsi_name,
                "prob_threshold": self.prob_thres,
                "inflam_cell_coords_path": str(json_detection_path),
            }

        # If not cached, run detection without visualization
        output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Running inflammatory cell detection as the file does not exist or force flag is set.")
        run_inflammatory_cell_detection(
            wsi_path=wsi_path,
            output_dir=output_dir,
            model_path="checkpoints/detection/",
            visualise=False  # Always false, we'll use separate visualization
        )

        # verify after detection
        if not json_detection_path.exists():
            self.logger.error(f"Detection output file not found at: {json_detection_path}")
            raise FileNotFoundError(f"Inflammatory cell detection JSON file not found at {json_detection_path}")

        with open(json_detection_path) as f:
            inflam = json.load(f)

        # Load all points first
        all_points = inflam["points"]
        total_points = len(all_points)
        
        # Filter points based on probability threshold
        filtered_points = [
            pt for pt in all_points 
            if "probability" in pt and pt["probability"] >= self.prob_thres
        ]
        
        # Extract coordinates from filtered points
        mm_coords = np.array(
            [[pt["point"][0], pt["point"][1]] for pt in filtered_points],
            dtype=np.float32,
        )
        
        self.logger.info(f"Filtered inflammatory cells from {total_points} to {len(filtered_points)} using threshold {self.prob_thres}")
        
        # Generate visualization if requested
        if visualise:
            self._run_detection_visualization(wsi_path, mm_coords, output_dir)
        
        return {
            "wsi_name": wsi_name,
            "prob_threshold": self.prob_thres,
            "inflam_cell_coords_path": str(json_detection_path),
        }
    
    def _run_detection_visualization(self, wsi_path, mm_coords, output_dir):
        """Generate detection visualization using the separate visualise_overlay.py"""
        try:
            from detection.visualise_overlay import generate_tiff_overlay, load_coordinates
            
            self.logger.info("Generating detection visualization using visualise_overlay.py")
            
            # Generate visualization
            output_path = output_dir / "inflammatory_cells_overlay.tiff"
            generate_tiff_overlay(mm_coords, wsi_path, output_path)
            
            self.logger.info(f"Detection visualization saved to {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate detection visualization: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def run_stage3(self, wsi_path: str, force: bool = False, visualise: bool = False) -> dict:
        import tifffile as tiff
        import pandas as pd
        
        self.logger.info("Running Stage 3: Quantification and Grading")
        self.logger.info("====== STAGE 3: GRADING STAGE STARTED ======")

        paths = self.get_output_paths(wsi_path)
        wsi_name = Path(wsi_path).stem
        
        # Check if grading already exists
        if paths["grading_report"].exists() and not force:
            try:
                with open(paths["grading_report"]) as f:
                    existing_results = json.load(f)
                
                # Check if parameters match
                existing_prob_thres = existing_results.get("prob_thres")
                
                if (existing_prob_thres == self.prob_thres):
                    self.logger.info(f"Grading already exists for parameters prob_thres={self.prob_thres}. Reusing results.")
                    
                    # If visualization is requested, generate it even with existing results
                    if visualise:
                        self.logger.info("Generating visualization for existing grading results")
                        try:
                            from grading.visualise_quantification import create_quantification_overlay
                            
                            # Load necessary data for visualization
                            tubule_mask_path = self.individual_reports_dir / wsi_name / "segmentation" / f"{wsi_name}_full_instance_mask_class1.tiff"
                            with tiff.TiffFile(tubule_mask_path) as tif:
                                tubule_mask = tif.asarray(out='memmap')
                                
                            # Load inflammatory cell detections
                            detection_json_path = self.individual_reports_dir / wsi_name / "detection" / "detected-inflammatory-cells.json"
                            with open(detection_json_path) as f:
                                inflam = json.load(f)
                            
                            # Filter points based on probability threshold
                            filtered_points = [
                                pt for pt in inflam["points"] 
                                if "probability" in pt and pt["probability"] >= self.prob_thres
                            ]
                            
                            # Extract coordinates (in mm)
                            mm_coords = np.array(
                                [[pt["point"][0], pt["point"][1]] for pt in filtered_points],
                                dtype=np.float32,
                            )
                            
                            # Load counts data for highlighting top tubules
                            counts_csv_path = Path(paths["counts_csv"])
                            existing_counts_df = None
                            if counts_csv_path.exists():
                                try:
                                    existing_counts_df = pd.read_csv(counts_csv_path)
                                except Exception as e:
                                    self.logger.warning(f"Could not load counts CSV for visualization: {e}")
                            
                            # Create visualization directory
                            vis_dir = Path(paths["counts_csv"]).parent / "visualization"
                            
                            # Generate overlay with top tubules highlighted
                            overlay_path = create_quantification_overlay(
                                wsi_path=wsi_path,
                                tubule_mask=tubule_mask,
                                cell_coords=mm_coords,
                                output_dir=vis_dir,
                                counts_df=existing_counts_df,
                                highlight_top_tubules=5,
                                highlight_style="circle"
                            )
                            
                            if overlay_path:
                                self.logger.info(f"Created quantification visualization at {overlay_path}")
                                existing_results["visualization_path"] = str(overlay_path)
                            
                        except Exception as e:
                            self.logger.error(f"Failed to create visualization: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                    
                    # Ensure grading_report path is included in return
                    existing_results["grading_report"] = str(paths["grading_report"])
                    
                    # Load quantification data to get missing fields like total_tubules, total_inflam_cells, etc.
                    try:
                        if paths["quant_json"].exists():
                            with open(paths["quant_json"]) as f:
                                quant_data = json.load(f)
                            # Merge quantification data with existing results
                            existing_results.update(quant_data)
                    except Exception as e:
                        self.logger.warning(f"Could not load quantification data: {e}")
                    
                    return existing_results
                else:
                    self.logger.info(f"Existing grading found but parameters differ (existing: prob_thres={existing_prob_thres}, current: prob_thres={self.prob_thres}). Recomputing.")
            except Exception as e:
                self.logger.warning(f"Error reading existing grading: {e}. Recomputing.")
        
        # Load data from shared segmentation directory
        tubule_mask_path = self.individual_reports_dir / wsi_name / "segmentation" / f"{wsi_name}_full_instance_mask_class1.tiff"
        if not tubule_mask_path.exists():
            raise FileNotFoundError(f"Required tubule mask not found at {tubule_mask_path}. Run Stage 1 first.")
        
        # Load data from shared detection directory
        detection_json_path = self.individual_reports_dir / wsi_name / "detection" / "detected-inflammatory-cells.json"
        if not detection_json_path.exists():
            raise FileNotFoundError(f"Required detection file not found at {detection_json_path}. Run Stage 2 first.")

        # Load the instance mask, which is memorymapped for large files
        with tiff.TiffFile(tubule_mask_path) as tif:
            tubule_mask = tif.asarray(out='memmap')

        # Load inflammatory cell detections
        with open(detection_json_path) as f:
            inflam = json.load(f)

        # Load all points first
        all_points = inflam["points"]
        total_points = len(all_points)
        
        # Filter points based on probability threshold
        filtered_points = [
            pt for pt in all_points 
            if "probability" in pt and pt["probability"] >= self.prob_thres
        ]
        
        # Extract coordinates from filtered points
        mm_coords = np.array(
            [[pt["point"][0], pt["point"][1]] for pt in filtered_points],
            dtype=np.float32,
        )
        
        self.logger.info(f"Filtered inflammatory cells from {total_points} to {len(filtered_points)} using threshold {self.prob_thres}")

        MICRONS_PER_PIXEL = 0.24199951445730394
        # Fix coordinate system: mm_coords are (x,y) but count_cells_in_tubules expects (y,x)
        # So we need to swap and convert: (x,y) -> (y,x) in pixels
        cell_coords = np.column_stack([
            (mm_coords[:, 1] * 1000 / MICRONS_PER_PIXEL).astype(np.int32),  # y coordinates
            (mm_coords[:, 0] * 1000 / MICRONS_PER_PIXEL).astype(np.int32)   # x coordinates
        ])

        # Debug logging
        self.logger.info(f"[DEBUG] Total inflammatory cells detected: {total_points}")
        self.logger.info(f"[DEBUG] Cells after prob_thres={self.prob_thres} filtering: {len(filtered_points)}")
        if len(filtered_points) > 0:
            self.logger.info(f"[DEBUG] MM coordinate range: X=[{mm_coords[:, 0].min():.3f}, {mm_coords[:, 0].max():.3f}], Y=[{mm_coords[:, 1].min():.3f}, {mm_coords[:, 1].max():.3f}]")
            self.logger.info(f"[DEBUG] Pixel coordinate range: X=[{cell_coords[:, 1].min()}, {cell_coords[:, 1].max()}], Y=[{cell_coords[:, 0].min()}, {cell_coords[:, 0].max()}]")
        
        self.logger.info(f"[DEBUG] Tubule mask shape: {tubule_mask.shape}")
        unique_tubule_ids = np.unique(tubule_mask)
        self.logger.info(f"[DEBUG] Unique tubule IDs: {len(unique_tubule_ids)} (range: {unique_tubule_ids.min()} to {unique_tubule_ids.max()})")
        
        from tiffslide import TiffSlide
        slide = TiffSlide(wsi_path)
        self.logger.info(f"[Stage 3] WSI name: {paths['wsi_name']}")
        self.logger.info(f"[Stage 3] WSI size: {slide.dimensions}")
        self.logger.info(f"[Stage 3] Tubule mask shape: {tubule_mask.shape}")
        self.logger.info(f"[Stage 3] Parameters: prob_thres={self.prob_thres}")

        counts_df = count_cells_in_tubules(cell_coords, tubule_mask)
        
        # Debug the results
        self.logger.info(f"[DEBUG] Count results: {len(counts_df)} tubules with cells")
        if len(counts_df) > 0:
            self.logger.info(f"[DEBUG] Top 5 tubules by cell count: {counts_df.head()[['tubule_id', 'cell_count']].to_dict('records')}")
        else:
            self.logger.warning("[DEBUG] No cells found in any tubules!")

        # Create grading directory only when we're actually saving results
        grading_dir = Path(paths["counts_csv"]).parent
        grading_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results directly without using save_counts_csv to avoid redundant files
        self.logger.info("Saving counts and computing summary statistics...")
        
        # Save the CSV and get summary stats
        counts_df.to_csv(paths["counts_csv"], index=False)
        summary_stats = analyze_tubule_cell_distribution(counts_df)
        
        self.logger.info(f"Saved cell counts to {paths['counts_csv']}")

        # Create visualization if requested
        visualization_path = None
        if visualise:
            self.logger.info("Generating visualization for quantification results")
            try:
                from grading.visualise_quantification import create_quantification_overlay
                
                # Create visualization directory
                vis_dir = grading_dir / "visualization"
                
                # Generate overlay with top tubules highlighted
                visualization_path = create_quantification_overlay(
                    wsi_path=wsi_path,
                    tubule_mask=tubule_mask,
                    cell_coords=mm_coords,
                    output_dir=vis_dir,
                    counts_df=counts_df,
                    highlight_top_tubules=5,
                    highlight_style="circle"
                )
                
                if visualization_path:
                    self.logger.info(f"Created quantification visualization at {visualization_path}")
            except Exception as e:
                self.logger.error(f"Failed to create visualization: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Create per_tubule_counts directly in the format needed for output
        per_tubule_counts = {int(k): int(v) for k, v in zip(counts_df['tubule_id'], counts_df['cell_count'])}

        # Calculate Banff tubulitis score
        grading_result = calculate_tubulitis_score(
            counts_df=counts_df,
            output_dir=grading_dir
        )

        # Check if we have ground truth T score
        true_t_score = None
        
        # Try to get ground truth from banff_scores.csv if it exists
        banff_csv = Path("banff_scores.csv")
        if banff_csv.exists():
            try:
                banff_df = pd.read_csv(banff_csv)
                
                # Try to find this WSI in the ground truth data
                wsi_filename = f"{wsi_name}.svs"
                match = banff_df[banff_df["filename"] == wsi_filename]
                
                if not match.empty and "T" in match.columns and pd.notna(match["T"].values[0]):
                    true_t_score = float(match["T"].values[0])
                    self.logger.info(f"Found ground truth T score: {true_t_score} for {wsi_name}")
            except Exception as e:
                self.logger.warning(f"Could not load ground truth T score: {e}")

        # Create quantification data (detailed cell counting results)
        quantification_data = {
            "wsi_name": paths["wsi_name"],
            "total_inflam_cells": int(len(cell_coords)),
            "total_tubules": int(len(np.unique(tubule_mask)) - 1),
            "mean_cells_per_tubule": float(summary_stats.get('mean_cells_per_tubule', 0.0)),
            "summary_stats": convert_numpy_types(summary_stats),
            "prob_thres": self.prob_thres,
            "param_tag": paths["param_tag"],
            "tubule_counts_csv": str(paths["counts_csv"]),
            "per_tubule_inflam_cell_counts": per_tubule_counts
        }
        
        # Add visualization path if available
        if visualization_path:
            quantification_data["visualization_path"] = str(visualization_path)

        # Create grading report (focused on scoring and evaluation)
        grading_report_data = {
            "wsi_name": paths["wsi_name"],
            "tubulitis_score_predicted": grading_result["score"],
            "prob_thres": self.prob_thres,
            "param_tag": paths["param_tag"],
            "quantification_json": str(paths["quant_json"])
        }
        
        # Add ground truth evaluation if available
        if true_t_score is not None:
            # Store ground truth in the same format as prediction (t0, t1, t2, t3)
            grading_report_data["tubulitis_score_ground_truth"] = f"t{int(round(true_t_score))}"
            
            # Extract numeric value from prediction for comparison
            pred_score = float(grading_result["score"][1:]) if grading_result["score"].startswith("t") else float(grading_result["score"])
            
            # Calculate difference and correctness
            grading_report_data["score_difference"] = abs(pred_score - true_t_score)
            grading_report_data["correct_category"] = (round(pred_score) == round(true_t_score))

        # Save separate files
        with open(paths["quant_json"], "w") as f:
            json.dump(quantification_data, f, indent=2)

        with open(paths["grading_report"], "w") as f:
            json.dump(grading_report_data, f, indent=2)

        self.logger.info(f"Saved quantification data to {paths['quant_json']}")
        self.logger.info(f"Saved grading report to {paths['grading_report']}")
        self.logger.info("====== STAGE 3: GRADING COMPLETED ======")
        
        # Return combined data for pipeline use
        combined_result = {**quantification_data, **grading_report_data}
        combined_result["grading_report"] = str(paths["grading_report"])
        return combined_result

    def create_summary_files(self, update_summary: bool = True) -> None:
        """Create or update summary files with separate results for each probability threshold.
        
        Args:
            update_summary: Whether to update the main summary files
        """
        if not update_summary:
            return  # Nothing to do
            
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_absolute_error, mean_squared_error, confusion_matrix,
            classification_report, cohen_kappa_score
        )
        from scipy.stats import spearmanr, pearsonr
        
        # Collect all grading results
        results = []
        for wsi_dir in self.individual_reports_dir.iterdir():
            if not wsi_dir.is_dir():
                continue
            wsi_name = wsi_dir.name
            
            # Look for grading reports in the new parameter-specific structure
            grading_dir = wsi_dir / "grading"
            if not grading_dir.exists():
                continue
                
            # Find all parameter-specific grading reports
            for param_dir in grading_dir.iterdir():
                if not param_dir.is_dir():
                    continue
                grading_report_path = param_dir / "grading_report.json"
                if not grading_report_path.exists():
                    continue
                    
                try:
                    with open(grading_report_path) as f:
                        data = json.load(f)
                    result = {
                        "wsi_name": wsi_name,
                        "tubulitis_score_predicted": data.get("tubulitis_score_predicted", data.get("tubulitis_score", None)),
                        "tubulitis_score_ground_truth": data.get("tubulitis_score_ground_truth", None),
                        "prob_thres": data.get("prob_thres", None),
                        "param_tag": data.get("param_tag", param_dir.name)
                    }
                    
                    # Also read quantification data if available
                    quantification_path = param_dir / "quantification.json"
                    if quantification_path.exists():
                        try:
                            with open(quantification_path) as f:
                                quant_data = json.load(f)
                            # Add quantification metrics to result
                            result.update({
                                "mean_cells_per_tubule": quant_data.get("mean_cells_per_tubule", None),
                                "total_inflam_cells": quant_data.get("total_inflam_cells", None),
                                "total_tubules": quant_data.get("total_tubules", None),
                                "summary_stats": quant_data.get("summary_stats", {}),
                                "total_cells": quant_data.get("total_cells", None),
                                "std_cells_per_tubule": quant_data.get("std_cells_per_tubule", None),
                                "max_cells_in_tubule": quant_data.get("max_cells_in_tubule", None),
                                "mean_cells_top_1pct": quant_data.get("mean_cells_top_1pct", None),
                                "mean_cells_top_5pct": quant_data.get("mean_cells_top_5pct", None),
                                "mean_cells_top_10pct": quant_data.get("mean_cells_top_10pct", None)
                            })
                            # Add summary stats fields as top-level columns for easier access
                            if "summary_stats" in quant_data:
                                stats = quant_data["summary_stats"]
                                result.update({
                                    "total_cells": stats.get("total_cells", None),
                                    "std_cells_per_tubule": stats.get("std_cells_per_tubule", None),
                                    "max_cells_in_tubule": stats.get("max_cells_in_tubule", None),
                                    "mean_cells_top_1pct": stats.get("mean_cells_top_1pct", None),
                                    "mean_cells_top_5pct": stats.get("mean_cells_top_5pct", None),
                                    "mean_cells_top_10pct": stats.get("mean_cells_top_10pct", None)
                                })
                        except Exception as e:
                            self.logger.warning(f"Could not read quantification data from {quantification_path}: {e}")
                    
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error reading {grading_report_path}: {e}")
        
        if not results:
            self.logger.warning("No grading results found to create summary files")
            return
            
        # Create the summary DataFrame
        results_df = pd.DataFrame(results)
        
        # Convert string T-scores to numeric values for calculations
        for col in ['tubulitis_score_predicted', 'tubulitis_score_ground_truth']:
            if col in results_df.columns:
                # Create a numeric version of the tubulitis score
                results_df[f'{col}_numeric'] = results_df[col].apply(
                    lambda x: float(x[1:]) if isinstance(x, str) and x.startswith('t') else x
                )
                
                # Handle any remaining non-numeric values
                results_df[f'{col}_numeric'] = pd.to_numeric(
                    results_df[f'{col}_numeric'], 
                    errors='coerce'  # Convert errors to NaN
                )
        
        # Create main summary directory
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined raw data (for backwards compatibility)
        scores_csv = self.summary_dir / "aggregated_scores_all_thresholds.csv"
        results_df.to_csv(scores_csv, index=False)
        self.logger.info(f"Saved combined aggregated scores to {scores_csv}")
        
        # Set up main versioned directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_version_dir = self.summary_dir / f"version_{timestamp}_combined"
        main_version_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(main_version_dir / "aggregated_scores_all_thresholds.csv", index=False)
        
        # Filter to only include results with ground truth for evaluation
        eval_df = results_df.dropna(subset=['tubulitis_score_ground_truth_numeric', 'tubulitis_score_predicted_numeric'])
        
        if len(eval_df) == 0:
            self.logger.warning("No results with ground truth available for evaluation")
            return
            
        self.logger.info(f"Evaluating {len(eval_df)} results with ground truth data")
        
        # Group by probability threshold and create separate summaries
        param_groups = eval_df.groupby('prob_thres')
        
        # Store metrics for all thresholds (for comparison)
        all_metrics_data = []
        
        # Process each probability threshold separately
        for prob_thres, group in param_groups:
            # Extract the actual float value if prob_thres is a tuple  
            if isinstance(prob_thres, tuple):
                prob_thres = prob_thres[0]
                
            self.logger.info(f"Creating summary for probability threshold: {prob_thres}")
            
            # Create separate directory for this threshold
            threshold_dir = self.summary_dir / f"prob_thres_{prob_thres}"
            threshold_dir.mkdir(parents=True, exist_ok=True)
            
            # Create versioned directory for this threshold
            threshold_version_dir = threshold_dir / f"version_{timestamp}"
            threshold_version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save threshold-specific data
            threshold_csv = threshold_dir / "aggregated_scores.csv"
            group.to_csv(threshold_csv, index=False)
            group.to_csv(threshold_version_dir / "aggregated_scores.csv", index=False)
            self.logger.info(f"Saved threshold-specific scores to {threshold_csv}")
            
            # Calculate metrics for this threshold
            y_true = group['tubulitis_score_ground_truth_numeric'].round().astype(int)
            y_pred = group['tubulitis_score_predicted_numeric'].round().astype(int)
            
            # Calculate standard classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Calculate regression metrics
            mae = mean_absolute_error(group['tubulitis_score_ground_truth_numeric'], group['tubulitis_score_predicted_numeric'])
            mse = mean_squared_error(group['tubulitis_score_ground_truth_numeric'], group['tubulitis_score_predicted_numeric'])
            rmse = np.sqrt(mse)
            
            # Calculate correlation coefficients - needs at least 2 samples
            if len(group) >= 2:
                try:
                    pearson_corr, pearson_p = pearsonr(group['tubulitis_score_ground_truth_numeric'], group['tubulitis_score_predicted_numeric'])
                    spearman_corr, spearman_p = spearmanr(group['tubulitis_score_ground_truth_numeric'], group['tubulitis_score_predicted_numeric'])
                except Exception as e:
                    self.logger.warning(f"Could not calculate correlation for threshold {prob_thres}: {e}")
                    pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
            else:
                self.logger.warning(f"Threshold {prob_thres} has only {len(group)} samples - skipping correlation calculation")
                pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
            
            # Calculate Cohen's Kappa (agreement metric)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # Calculate custom metrics
            # Accuracy within 1 score point
            within_one = np.mean(np.abs(group['tubulitis_score_ground_truth_numeric'] - group['tubulitis_score_predicted_numeric']) <= 1)
            
            # Weighted accuracy (penalizes larger errors more)
            weights = 1.0 / (1.0 + np.abs(group['tubulitis_score_ground_truth_numeric'] - group['tubulitis_score_predicted_numeric']))
            weighted_accuracy = np.mean(weights)
            
            # Calculate additional comprehensive metrics
            from sklearn.metrics import f1_score as sklearn_f1_score
            
            # F1 scores (macro, micro, weighted)
            f1_macro = sklearn_f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_micro = sklearn_f1_score(y_true, y_pred, average='micro', zero_division=0)  
            f1_weighted = sklearn_f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Class-specific accuracy (per-class recall)
            from sklearn.metrics import classification_report
            clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Extract class-specific accuracies (recall for each class)
            class_accuracies = {}
            for i in range(4):  # t0, t1, t2, t3
                class_key = str(i)
                if class_key in clf_report:
                    class_accuracies[f'class_{i}_accuracy'] = clf_report[class_key]['recall']
                else:
                    class_accuracies[f'class_{i}_accuracy'] = 0.0
            
            # Quadratic weighted kappa
            try:
                from sklearn.metrics import cohen_kappa_score
                kappa_quadratic = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            except Exception as e:
                self.logger.warning(f"Could not calculate quadratic weighted kappa: {e}")
                kappa_quadratic = np.nan
            
            # Exact accuracy (already calculated as 'accuracy' above)
            exact_accuracy = accuracy
            
            threshold_metrics = {
                'prob_thres': prob_thres,
                'sample_count': len(group),
                # Basic metrics
                'exact_accuracy': exact_accuracy,
                'within_1_grade_accuracy': within_one,
                'weighted_accuracy': weighted_accuracy,
                # Classification metrics  
                'precision_weighted': precision,
                'recall_weighted': recall,
                'f1_score_weighted': f1_weighted,
                'f1_score_macro': f1_macro,
                'f1_score_micro': f1_micro,
                # Regression metrics
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                # Correlation coefficients
                'pearson_corr': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p_value': spearman_p,
                # Agreement metrics
                'cohens_kappa': kappa,
                'quadratic_weighted_kappa': kappa_quadratic,
                # Class-specific accuracies
                **class_accuracies
            }
            
            # Create comprehensive text report
            report_lines = [
                f"COMPREHENSIVE EVALUATION METRICS REPORT",
                f"=====================================",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Probability Threshold: {prob_thres}",
                f"Number of Samples: {len(group)}",
                f"",
                f"ACCURACY METRICS",
                f"----------------",
                f"Exact Accuracy:           {exact_accuracy:.4f} ({exact_accuracy*100:.2f}%)",
                f"Within-1-Grade Accuracy:  {within_one:.4f} ({within_one*100:.2f}%)",
                f"Weighted Accuracy:        {weighted_accuracy:.4f} ({weighted_accuracy*100:.2f}%)",
                f"",
                f"F1 SCORES",
                f"---------",
                f"F1-Score (Weighted):      {f1_weighted:.4f}",
                f"F1-Score (Macro):         {f1_macro:.4f}",
                f"F1-Score (Micro):         {f1_micro:.4f}",
                f"",
                f"CLASS-SPECIFIC ACCURACY (Recall per class)",
                f"------------------------------------------"
            ]
            
            for i in range(4):
                class_acc = class_accuracies.get(f'class_{i}_accuracy', 0.0)
                n_true = np.sum(y_true == i)
                report_lines.append(f"Class t{i} Accuracy:       {class_acc:.4f} ({class_acc*100:.2f}%) [n={n_true}]")
            
            report_lines.extend([
                f"",
                f"REGRESSION METRICS",
                f"------------------",
                f"Mean Absolute Error:      {mae:.4f}",
                f"Mean Squared Error:       {mse:.4f}",
                f"Root Mean Squared Error:  {rmse:.4f}",
                f"",
                f"CORRELATION COEFFICIENTS",
                f"------------------------",
                f"Pearson Correlation:      {pearson_corr:.4f} (p={pearson_p:.4f})",
                f"Spearman Correlation:     {spearman_corr:.4f} (p={spearman_p:.4f})",
                f"",
                f"AGREEMENT METRICS",
                f"-----------------",
                f"Cohen's Kappa:            {kappa:.4f}",
                f"Quadratic Weighted Kappa: {kappa_quadratic:.4f}",
                f"",
                f"KAPPA INTERPRETATION",
                f"--------------------"
            ])
            
            # Add kappa interpretation
            def interpret_kappa(k):
                if np.isnan(k):
                    return "Not Available"
                elif k < 0:
                    return "Poor (worse than chance)"
                elif k <= 0.20:
                    return "Slight"
                elif k <= 0.40:
                    return "Fair"
                elif k <= 0.60:
                    return "Moderate"
                elif k <= 0.80:
                    return "Substantial"
                else:
                    return "Almost Perfect"
            
            report_lines.extend([
                f"Cohen's Kappa:            {interpret_kappa(kappa)}",
                f"Quadratic Weighted Kappa: {interpret_kappa(kappa_quadratic)}",
                f"",
                f"CONFUSION MATRIX",
                f"----------------"
            ])
            
            # Add confusion matrix to report
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
            report_lines.append("         Predicted")
            report_lines.append("       t0  t1  t2  t3")
            for i in range(4):
                line = f"True t{i}"
                for j in range(4):
                    line += f"{cm[i,j]:4d}"
                report_lines.append(line)
            
            report_lines.extend([
                f"",
                f"CLASSIFICATION REPORT",
                f"--------------------"
            ])
            
            # Add detailed classification report
            clf_report_str = classification_report(y_true, y_pred, 
                                                 target_names=['t0', 't1', 't2', 't3'],
                                                 zero_division=0)
            report_lines.extend(clf_report_str.split('\n'))
            
            # Save the comprehensive text report
            report_path = threshold_dir / "comprehensive_evaluation_report.txt"
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            report_version_path = threshold_version_dir / "comprehensive_evaluation_report.txt"
            with open(report_version_path, 'w') as f:
                f.write('\n'.join(report_lines))
                
            self.logger.info(f"Saved comprehensive evaluation report to {report_path}")
            
            # Save metrics for this threshold
            threshold_metrics_df = pd.DataFrame([threshold_metrics])
            metrics_path = threshold_dir / "evaluation_metrics.csv"
            threshold_metrics_df.to_csv(metrics_path, index=False)
            threshold_metrics_df.to_csv(threshold_version_dir / "evaluation_metrics.csv", index=False)
            self.logger.info(f"Saved threshold-specific metrics to {metrics_path}")
            
            # Add to overall metrics collection
            all_metrics_data.append(threshold_metrics)
            
            # Create threshold-specific visualizations
            self._create_threshold_visualizations(group, threshold_dir, threshold_version_dir, prob_thres)
        
        # Save combined metrics comparison (for comparing across thresholds)
        if all_metrics_data:
            all_metrics_df = pd.DataFrame(all_metrics_data)
            comparison_path = self.summary_dir / "threshold_comparison_metrics.csv"
            all_metrics_df.to_csv(comparison_path, index=False)
            all_metrics_df.to_csv(main_version_dir / "threshold_comparison_metrics.csv", index=False)
            self.logger.info(f"Saved threshold comparison metrics to {comparison_path}")
            
            # Create comparison visualizations
            self._create_threshold_comparison_visualizations(all_metrics_df, eval_df, self.summary_dir, main_version_dir)
        
        self.logger.info(f"Summary files created with separate folders for each probability threshold in {self.summary_dir}")


    def _create_threshold_visualizations(self, group_df, threshold_dir, version_dir, prob_thres):
        """Create visualizations for a specific probability threshold."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import confusion_matrix
        
        # 1. Predicted vs Ground Truth scatter plot
        plt.figure(figsize=(8, 6))
        
        x = group_df['tubulitis_score_ground_truth_numeric']
        y = group_df['tubulitis_score_predicted_numeric']
        
        # Add jitter to avoid overplotting
        x_jitter = x + np.random.normal(0, 0.05, len(x))
        y_jitter = y + np.random.normal(0, 0.05, len(y))
        
        plt.scatter(x_jitter, y_jitter, alpha=0.7, s=50)
        
        # Add perfect prediction line
        plt.plot([0, 3], [0, 3], 'k--', alpha=0.5, label='Perfect prediction')
        
        # Add regression line
        if len(x) >= 2 and x.nunique() > 1 and y.nunique() > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(np.sort(x), p(np.sort(x)), "r--", alpha=0.8, label='Trend line')
            
            # Calculate correlation
            try:
                pearson_corr, _ = pearsonr(x, y)
                corr_text = f'r={pearson_corr:.3f}'
            except:
                corr_text = 'r=N/A'
        else:
            corr_text = 'r=N/A'
        
        plt.title(f'Predicted vs Ground Truth (prob_thres={prob_thres}, {corr_text})')
        plt.xlabel('Ground Truth T-score')
        plt.ylabel('Predicted T-score')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        pred_vs_truth_path = threshold_dir / "predicted_vs_ground_truth.png"
        plt.savefig(pred_vs_truth_path)
        plt.savefig(version_dir / "predicted_vs_ground_truth.png")
        plt.close()
        
        # 2. Confusion Matrix
        if len(group_df) > 1:
            plt.figure(figsize=(6, 5))
            
            y_true = group_df['tubulitis_score_ground_truth_numeric'].round().astype(int)
            y_pred = group_df['tubulitis_score_predicted_numeric'].round().astype(int)
            
            try:
                unique_classes = np.unique(np.concatenate([y_true, y_pred]))
                if len(unique_classes) > 1:
                    cm = confusion_matrix(y_true, y_pred)
                    
                    # Normalize confusion matrix
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        cm_norm = np.nan_to_num(cm_norm, nan=0)
                    
                    # Create heatmap
                    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                               xticklabels=[f't{i}' for i in range(4)],
                               yticklabels=[f't{i}' for i in range(4)])
                    
                    plt.title(f'Confusion Matrix (prob_thres={prob_thres})')
                    plt.xlabel('Predicted T-score')
                    plt.ylabel('Ground Truth T-score')
                    
                    cm_path = threshold_dir / "confusion_matrix.png"
                    plt.savefig(cm_path)
                    plt.savefig(version_dir / "confusion_matrix.png")
                    plt.close()
            except Exception as e:
                self.logger.warning(f"Error creating confusion matrix for threshold {prob_thres}: {e}")
                plt.close()
        
        # 3. Error distribution
        plt.figure(figsize=(8, 5))
        
        error = np.abs(group_df['tubulitis_score_ground_truth_numeric'] - group_df['tubulitis_score_predicted_numeric'])
        
        plt.hist(error, bins=np.arange(0, 4.5, 0.5), alpha=0.7, edgecolor='black')
        plt.title(f'Prediction Error Distribution (prob_thres={prob_thres})')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        error_dist_path = threshold_dir / "error_distribution.png"
        plt.savefig(error_dist_path)
        plt.savefig(version_dir / "error_distribution.png")
        plt.close()
        
        # 4. Feature correlation analysis
        features = [
            'mean_cells_per_tubule', 'total_inflam_cells', 'total_tubules', 
            'total_cells', 'std_cells_per_tubule', 'max_cells_in_tubule',
            'mean_cells_top_1pct', 'mean_cells_top_5pct', 'mean_cells_top_10pct'
        ]
        
        correlation_data = []
        for feature in features:
            if feature in group_df.columns:
                try:
                    # Skip if there are fewer than 2 samples or all values are identical
                    if len(group_df) < 2 or group_df[feature].nunique() <= 1 or group_df['tubulitis_score_ground_truth_numeric'].nunique() <= 1:
                        self.logger.warning(f"Skipping correlation for feature {feature} in threshold {prob_thres}: insufficient data variation")
                        continue
                        
                    pearson_corr, pearson_p = pearsonr(
                        group_df[feature], 
                        group_df['tubulitis_score_ground_truth_numeric']
                    )
                    spearman_corr, spearman_p = spearmanr(
                        group_df[feature], 
                        group_df['tubulitis_score_ground_truth_numeric']
                    )
                    
                    correlation_data.append({
                        'feature': feature,
                        'pearson_corr': pearson_corr,
                        'pearson_p': pearson_p,
                        'spearman_corr': spearman_corr,
                        'spearman_p': spearman_p
                    })
                except Exception as e:
                    self.logger.warning(f"Could not calculate correlation for {feature} in threshold {prob_thres}: {e}")
        
        if correlation_data:
            correlation_df = pd.DataFrame(correlation_data)
            correlation_path = threshold_dir / "feature_correlations.csv"
            correlation_df.to_csv(correlation_path, index=False)
            correlation_df.to_csv(version_dir / "feature_correlations.csv", index=False)
            self.logger.info(f"Saved feature correlations to {correlation_path}")
            
            # Create feature correlation heatmap
            if len(correlation_data) > 0:
                plt.figure(figsize=(10, 6))
                
                # Filter to significant correlations
                sig_correlations = correlation_df[correlation_df['pearson_p'] < 0.05]
                
                if not sig_correlations.empty:
                    # Create bar plot of significant correlations
                    plt.barh(sig_correlations['feature'], sig_correlations['pearson_corr'])
                    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                    plt.title(f'Feature Correlation with Ground Truth T-score (prob_thres={prob_thres})')
                    plt.xlabel('Pearson Correlation')
                    plt.grid(alpha=0.3)
                    
                    feature_corr_path = threshold_dir / "feature_correlations.png"
                    plt.savefig(feature_corr_path)
                    plt.savefig(version_dir / "feature_correlations.png")
                else:
                    self.logger.warning(f"No significant correlations found for threshold {prob_thres}")
                plt.close()
        
        # 5. Feature correlation with prediction error
        group_df_with_error = group_df.copy()
        group_df_with_error['error'] = np.abs(group_df['tubulitis_score_predicted_numeric'] - group_df['tubulitis_score_ground_truth_numeric'])
        
        error_corr_data = []
        for feature in features:
            if feature in group_df_with_error.columns:
                try:
                    # Skip if there are fewer than 2 samples or all values are identical
                    if len(group_df_with_error) < 2 or group_df_with_error[feature].nunique() <= 1 or group_df_with_error['error'].nunique() <= 1:
                        self.logger.warning(f"Skipping error correlation for feature {feature} in threshold {prob_thres}: insufficient data variation")
                        continue
                        
                    pearson_corr, pearson_p = pearsonr(group_df_with_error[feature], group_df_with_error['error'])
                    error_corr_data.append({
                        'feature': feature,
                        'correlation': pearson_corr,
                        'p_value': pearson_p
                    })
                except Exception as e:
                    self.logger.warning(f"Could not calculate error correlation for {feature} in threshold {prob_thres}: {e}")
        
        if error_corr_data:
            error_corr_df = pd.DataFrame(error_corr_data)
            error_corr_df = error_corr_df.sort_values('correlation', ascending=False)
            
            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.barh(error_corr_df['feature'], error_corr_df['correlation'])
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            plt.title(f'Feature Correlation with Prediction Error (prob_thres={prob_thres})')
            plt.xlabel('Pearson Correlation')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            error_corr_path = threshold_dir / "error_correlations.png"
            plt.savefig(error_corr_path)
            plt.savefig(version_dir / "error_correlations.png")
            
            # Save to CSV
            error_corr_df.to_csv(threshold_dir / "error_correlations.csv", index=False)
            error_corr_df.to_csv(version_dir / "error_correlations.csv", index=False)
            plt.close()
        
        # 6. Mean cells per tubule correlation analysis
        if 'mean_cells_per_tubule' in group_df.columns:
            plt.figure(figsize=(15, 5))
            
            # Subplot 1: Correlation with ground truth T-score
            plt.subplot(1, 3, 1)
            sns.regplot(
                data=group_df, 
                x='tubulitis_score_ground_truth_numeric',
                y='mean_cells_per_tubule', 
                scatter_kws={'alpha': 0.7, 's': 50},
                line_kws={'color': 'red'}
            )
            
            # Calculate correlation
            try:
                if len(group_df) >= 2 and group_df['mean_cells_per_tubule'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        group_df['tubulitis_score_ground_truth_numeric'],
                        group_df['mean_cells_per_tubule']
                    )
                    plt.title(f'Ground Truth vs Mean Cells/Tubule\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                else:
                    plt.title('Ground Truth vs Mean Cells/Tubule\n(Insufficient data variation)')
            except Exception as e:
                self.logger.warning(f"Error calculating correlation: {e}")
                plt.title('Ground Truth vs Mean Cells/Tubule')
            
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Mean Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Correlation with predicted T-score
            plt.subplot(1, 3, 2)
            sns.regplot(
                data=group_df, 
                x='tubulitis_score_predicted_numeric',
                y='mean_cells_per_tubule', 
                scatter_kws={'alpha': 0.7, 's': 50},
                line_kws={'color': 'green'}
            )
            
            try:
                if len(group_df) >= 2 and group_df['mean_cells_per_tubule'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        group_df['tubulitis_score_predicted_numeric'],
                        group_df['mean_cells_per_tubule']
                    )
                    plt.title(f'Predicted vs Mean Cells/Tubule\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                else:
                    plt.title('Predicted vs Mean Cells/Tubule')
            except Exception as e:
                plt.title('Predicted vs Mean Cells/Tubule')
            
            plt.xlabel('Predicted T-score')
            plt.ylabel('Mean Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Distribution by T-score category
            plt.subplot(1, 3, 3)
            
            # Ensure T-score is formatted correctly for the box plot
            group_df_copy = group_df.copy()
            group_df_copy['t_score_category'] = group_df_copy['tubulitis_score_ground_truth_numeric'].round().astype(int).map(lambda x: f't{x}')
            
            # Create box plot
            sns.boxplot(data=group_df_copy, x='t_score_category', y='mean_cells_per_tubule')
            plt.title(f'Distribution of Mean Cells per Tubule by T-score (prob_thres={prob_thres})')
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Mean Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plots
            cells_vs_tscore_path = threshold_dir / "mean_cells_vs_tscore.png"
            plt.savefig(cells_vs_tscore_path)
            plt.savefig(version_dir / "mean_cells_vs_tscore.png")
            
            # Save the data points for this specific relationship
            cells_vs_tscore_data = group_df[['wsi_name', 'mean_cells_per_tubule', 
                                           'tubulitis_score_ground_truth', 
                                           'tubulitis_score_ground_truth_numeric',
                                           'tubulitis_score_predicted',
                                           'prob_thres']]
            cells_vs_tscore_path_csv = threshold_dir / "mean_cells_vs_tscore.csv"
            cells_vs_tscore_data.to_csv(cells_vs_tscore_path_csv, index=False)
            cells_vs_tscore_data.to_csv(version_dir / "mean_cells_vs_tscore.csv", index=False)
            
            self.logger.info(f"Saved mean cells per tubule vs T-score analysis to {cells_vs_tscore_path}")
            plt.close()
        
        # 7. Max cells per tubule correlation analysis
        if 'max_cells_in_tubule' in group_df.columns:
            plt.figure(figsize=(15, 5))
            
            # Subplot 1: Correlation with ground truth T-score
            plt.subplot(1, 3, 1)
            sns.regplot(
                data=group_df, 
                x='tubulitis_score_ground_truth_numeric',
                y='max_cells_in_tubule', 
                scatter_kws={'alpha': 0.7, 's': 50},
                line_kws={'color': 'red'}
            )
            
            # Calculate correlation
            try:
                if len(group_df) >= 2 and group_df['max_cells_in_tubule'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        group_df['tubulitis_score_ground_truth_numeric'],
                        group_df['max_cells_in_tubule']
                    )
                    plt.title(f'Ground Truth vs Max Cells/Tubule\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                else:
                    plt.title('Ground Truth vs Max Cells/Tubule\n(Insufficient data variation)')
            except Exception as e:
                self.logger.warning(f"Error calculating max cells correlation: {e}")
                plt.title('Ground Truth vs Max Cells/Tubule')
            
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Max Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Correlation with predicted T-score
            plt.subplot(1, 3, 2)
            sns.regplot(
                data=group_df, 
                x='tubulitis_score_predicted_numeric',
                y='max_cells_in_tubule', 
                scatter_kws={'alpha': 0.7, 's': 50},
                line_kws={'color': 'green'}
            )
            
            try:
                if len(group_df) >= 2 and group_df['max_cells_in_tubule'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        group_df['tubulitis_score_predicted_numeric'],
                        group_df['max_cells_in_tubule']
                    )
                    plt.title(f'Predicted vs Max Cells/Tubule\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                else:
                    plt.title('Predicted vs Max Cells/Tubule')
            except Exception as e:
                plt.title('Predicted vs Max Cells/Tubule')
            
            plt.xlabel('Predicted T-score')
            plt.ylabel('Max Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Distribution by T-score category
            plt.subplot(1, 3, 3)
            group_df_copy = group_df.copy()
            group_df_copy['t_score_category'] = group_df_copy['tubulitis_score_ground_truth_numeric'].round().astype(int).map(lambda x: f't{x}')
            sns.boxplot(data=group_df_copy, x='t_score_category', y='max_cells_in_tubule')
            plt.title('Max Cells/Tubule Distribution by T-score')
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Max Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            max_cells_corr_path = threshold_dir / "max_cells_correlation_analysis.png"
            plt.savefig(max_cells_corr_path)
            plt.savefig(version_dir / "max_cells_correlation_analysis.png")
            plt.close()
            
            self.logger.info(f"Saved max cells correlation analysis to {max_cells_corr_path}")
        
        # 8. Total inflammatory cells correlation analysis
        if 'total_inflam_cells' in group_df.columns:
            plt.figure(figsize=(15, 5))
            
            # Subplot 1: Correlation with ground truth T-score
            plt.subplot(1, 3, 1)
            sns.regplot(
                data=group_df, 
                x='tubulitis_score_ground_truth_numeric',
                y='total_inflam_cells', 
                scatter_kws={'alpha': 0.7, 's': 50},
                line_kws={'color': 'red'}
            )
            
            # Calculate correlation
            try:
                if len(group_df) >= 2 and group_df['total_inflam_cells'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        group_df['tubulitis_score_ground_truth_numeric'],
                        group_df['total_inflam_cells']
                    )
                    plt.title(f'Ground Truth vs Total Inflammatory Cells\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                else:
                    plt.title('Ground Truth vs Total Inflammatory Cells\n(Insufficient data variation)')
            except Exception as e:
                self.logger.warning(f"Error calculating total cells correlation: {e}")
                plt.title('Ground Truth vs Total Inflammatory Cells')
            
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Total Inflammatory Cells')
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Correlation with predicted T-score
            plt.subplot(1, 3, 2)
            sns.regplot(
                data=group_df, 
                x='tubulitis_score_predicted_numeric',
                y='total_inflam_cells', 
                scatter_kws={'alpha': 0.7, 's': 50},
                line_kws={'color': 'green'}
            )
            
            try:
                if len(group_df) >= 2 and group_df['total_inflam_cells'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        group_df['tubulitis_score_predicted_numeric'],
                        group_df['total_inflam_cells']
                    )
                    plt.title(f'Predicted vs Total Inflammatory Cells\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                else:
                    plt.title('Predicted vs Total Inflammatory Cells')
            except Exception as e:
                plt.title('Predicted vs Total Inflammatory Cells')
            
            plt.xlabel('Predicted T-score')
            plt.ylabel('Total Inflammatory Cells')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Distribution by T-score category
            plt.subplot(1, 3, 3)
            group_df_copy = group_df.copy()
            group_df_copy['t_score_category'] = group_df_copy['tubulitis_score_ground_truth_numeric'].round().astype(int).map(lambda x: f't{x}')
            sns.boxplot(data=group_df_copy, x='t_score_category', y='total_inflam_cells')
            plt.title('Total Inflammatory Cells Distribution by T-score')
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Total Inflammatory Cells')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            total_cells_corr_path = threshold_dir / "total_cells_correlation_analysis.png"
            plt.savefig(total_cells_corr_path)
            plt.savefig(version_dir / "total_cells_correlation_analysis.png")
            plt.close()
            
            self.logger.info(f"Saved total inflammatory cells correlation analysis to {total_cells_corr_path}")
        
        # 9. Specific analysis of mean inflammatory cells per tubule vs tubulitis score
        if 'mean_cells_per_tubule' in group_df.columns:
            plt.figure(figsize=(12, 10))
            
            # Create scatter plot with regression line
            plt.subplot(2, 1, 1)
            sns.regplot(
                data=group_df, 
                x='tubulitis_score_ground_truth_numeric',
                y='mean_cells_per_tubule', 
                scatter_kws={'alpha': 0.7},
                line_kws={'color': 'red'}
            )
            
            # Calculate correlation
            try:
                if len(group_df) >= 2 and group_df['mean_cells_per_tubule'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        group_df['tubulitis_score_ground_truth_numeric'],
                        group_df['mean_cells_per_tubule']
                    )
                    spearman_r, spearman_p = spearmanr(
                        group_df['tubulitis_score_ground_truth_numeric'], 
                        group_df['mean_cells_per_tubule']
                    )
                    
                    plt.title(f'Ground Truth vs Mean Cells/Tubule (prob_thres={prob_thres})\nPearson r={pearson_r:.3f} (p={p_value:.4f}), '
                             f'Spearman r={spearman_r:.3f} (p={spearman_p:.4f})')
                else:
                    plt.title(f'Ground Truth vs Mean Cells/Tubule (prob_thres={prob_thres})\n'
                             'Not enough data variation for correlation calculation')
            except Exception as e:
                self.logger.warning(f"Error calculating correlation for mean_cells_per_tubule: {e}")
                plt.title(f'Ground Truth vs Mean Cells/Tubule (prob_thres={prob_thres})')
            
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Mean Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            # Create box plot to show distribution by T-score
            plt.subplot(2, 1, 2)
            
            # Ensure T-score is formatted correctly for the box plot
            group_df_copy = group_df.copy()
            group_df_copy['t_score_category'] = group_df_copy['tubulitis_score_ground_truth_numeric'].round().astype(int).map(lambda x: f't{x}')
            
            # Create box plot
            sns.boxplot(data=group_df_copy, x='t_score_category', y='mean_cells_per_tubule')
            plt.title(f'Distribution of Mean Cells per Tubule by T-score (prob_thres={prob_thres})')
            plt.xlabel('Ground Truth T-score')
            plt.ylabel('Mean Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plots
            cells_vs_tscore_path = threshold_dir / "mean_cells_vs_tscore.png"
            plt.savefig(cells_vs_tscore_path)
            plt.savefig(version_dir / "mean_cells_vs_tscore.png")
            
            # Save the data points for this specific relationship
            cells_vs_tscore_data = group_df[['wsi_name', 'mean_cells_per_tubule', 
                                           'tubulitis_score_ground_truth', 
                                           'tubulitis_score_ground_truth_numeric',
                                           'tubulitis_score_predicted',
                                           'prob_thres']]
            cells_vs_tscore_path_csv = threshold_dir / "mean_cells_vs_tscore.csv"
            cells_vs_tscore_data.to_csv(cells_vs_tscore_path_csv, index=False)
            cells_vs_tscore_data.to_csv(version_dir / "mean_cells_vs_tscore.csv", index=False)
            
            self.logger.info(f"Saved mean cells per tubule vs T-score analysis to {cells_vs_tscore_path}")
            plt.close()
        
        # 10. Top percentile correlation analysis
        top_percentile_features = ['mean_cells_top_1pct', 'mean_cells_top_5pct', 'mean_cells_top_10pct']
        
        for feature in top_percentile_features:
            if feature in group_df.columns:
                plt.figure(figsize=(15, 5))
                
                # Extract percentile from feature name for titles
                pct_name = feature.replace('mean_cells_top_', '').replace('pct', '%')
                
                # Subplot 1: Correlation with ground truth T-score
                plt.subplot(1, 3, 1)
                sns.regplot(
                    data=group_df, 
                    x='tubulitis_score_ground_truth_numeric',
                    y=feature, 
                    scatter_kws={'alpha': 0.7, 's': 50},
                    line_kws={'color': 'red'}
                )
                
                # Calculate correlation
                try:
                    if len(group_df) >= 2 and group_df[feature].nunique() > 1:
                        pearson_r, p_value = pearsonr(
                            group_df['tubulitis_score_ground_truth_numeric'],
                            group_df[feature]
                        )
                        plt.title(f'Ground Truth vs Top {pct_name} Mean Cells\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                    else:
                        plt.title(f'Ground Truth vs Top {pct_name} Mean Cells\n(Insufficient data variation)')
                except Exception as e:
                    self.logger.warning(f"Error calculating correlation for {feature}: {e}")
                    plt.title(f'Ground Truth vs Top {pct_name} Mean Cells')
                
                plt.xlabel('Ground Truth T-score')
                plt.ylabel(f'Mean Cells in Top {pct_name} Tubules')
                plt.grid(True, alpha=0.3)
                
                # Subplot 2: Correlation with predicted T-score
                plt.subplot(1, 3, 2)
                sns.regplot(
                    data=group_df, 
                    x='tubulitis_score_predicted_numeric',
                    y=feature, 
                    scatter_kws={'alpha': 0.7, 's': 50},
                    line_kws={'color': 'green'}
                )
                
                try:
                    if len(group_df) >= 2 and group_df[feature].nunique() > 1:
                        pearson_r, p_value = pearsonr(
                            group_df['tubulitis_score_predicted_numeric'],
                            group_df[feature]
                        )
                        plt.title(f'Predicted vs Top {pct_name} Mean Cells\nPearson r={pearson_r:.3f} (p={p_value:.4f})')
                    else:
                        plt.title(f'Predicted vs Top {pct_name} Mean Cells')
                except Exception as e:
                    plt.title(f'Predicted vs Top {pct_name} Mean Cells')
                
                plt.xlabel('Predicted T-score')
                plt.ylabel(f'Mean Cells in Top {pct_name} Tubules')
                plt.grid(True, alpha=0.3)
                
                # Subplot 3: Distribution by T-score category
                plt.subplot(1, 3, 3)
                group_df_copy = group_df.copy()
                group_df_copy['t_score_category'] = group_df_copy['tubulitis_score_ground_truth_numeric'].round().astype(int).map(lambda x: f't{x}')
                sns.boxplot(data=group_df_copy, x='t_score_category', y=feature)
                plt.title(f'Top {pct_name} Mean Cells Distribution by T-score')
                plt.xlabel('Ground Truth T-score')
                plt.ylabel(f'Mean Cells in Top {pct_name} Tubules')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                feature_safe_name = feature.replace('_', '')
                feature_corr_path = threshold_dir / f"{feature_safe_name}_correlation_analysis.png"
                plt.savefig(feature_corr_path)
                plt.savefig(version_dir / f"{feature_safe_name}_correlation_analysis.png")
                plt.close()
                
                # Save the data points for this specific relationship
                feature_data = group_df[['wsi_name', feature, 
                                       'tubulitis_score_ground_truth', 
                                       'tubulitis_score_ground_truth_numeric',
                                       'tubulitis_score_predicted',
                                       'prob_thres']]
                feature_data_path = threshold_dir / f"{feature_safe_name}_vs_tscore.csv"
                feature_data.to_csv(feature_data_path, index=False)
                feature_data.to_csv(version_dir / f"{feature_safe_name}_vs_tscore.csv", index=False)
                
                self.logger.info(f"Saved {feature} correlation analysis to {feature_corr_path}")
        
        # 11. Combined top percentiles correlation comparison
        available_top_features = [f for f in top_percentile_features if f in group_df.columns]
        
        if available_top_features and len(group_df) >= 2:
            plt.figure(figsize=(15, 10))
            
            # Create scatter plots for all top percentile features
            n_features = len(available_top_features)
            for i, feature in enumerate(available_top_features):
                pct_name = feature.replace('mean_cells_top_', '').replace('pct', '%')
                
                # Top row: vs Ground Truth
                plt.subplot(2, n_features, i + 1)
                sns.regplot(
                    data=group_df, 
                    x='tubulitis_score_ground_truth_numeric',
                    y=feature, 
                    scatter_kws={'alpha': 0.7, 's': 30},
                    line_kws={'color': 'red'}
                )
                
                try:
                    if group_df[feature].nunique() > 1:
                        pearson_r, p_value = pearsonr(
                            group_df['tubulitis_score_ground_truth_numeric'],
                            group_df[feature]
                        )
                        plt.title(f'Top {pct_name}\nr={pearson_r:.3f} (p={p_value:.3f})')
                    else:
                        plt.title(f'Top {pct_name}\n(No variation)')
                except Exception as e:
                    plt.title(f'Top {pct_name}')
                
                plt.xlabel('Ground Truth T-score')
                plt.ylabel(f'Mean Cells (Top {pct_name})')
                plt.grid(True, alpha=0.3)
                
                # Bottom row: vs Predicted
                plt.subplot(2, n_features, i + 1 + n_features)
                sns.regplot(
                    data=group_df, 
                    x='tubulitis_score_predicted_numeric',
                    y=feature, 
                    scatter_kws={'alpha': 0.7, 's': 30},
                    line_kws={'color': 'green'}
                )
                
                try:
                    if group_df[feature].nunique() > 1:
                        pearson_r, p_value = pearsonr(
                            group_df['tubulitis_score_predicted_numeric'],
                            group_df[feature]
                        )
                        plt.title(f'Top {pct_name}\nr={pearson_r:.3f} (p={p_value:.3f})')
                    else:
                        plt.title(f'Top {pct_name}\n(No variation)')
                except Exception as e:
                    plt.title(f'Top {pct_name}')
                
                plt.xlabel('Predicted T-score')
                plt.ylabel(f'Mean Cells (Top {pct_name})')
                plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'Top Percentile Features Correlation Comparison (prob_thres={prob_thres})', fontsize=14)
            plt.tight_layout()
            
            # Save combined plot
            combined_top_path = threshold_dir / "top_percentiles_correlation_comparison.png"
            plt.savefig(combined_top_path)
            plt.savefig(version_dir / "top_percentiles_correlation_comparison.png")
            plt.close()
            
            self.logger.info(f"Saved combined top percentiles correlation comparison to {combined_top_path}")
        
        # 12. Create a README for the threshold-specific version directory
        from datetime import datetime
        with open(version_dir / "README.txt", "w") as f:
            f.write(f"Threshold-specific analysis for prob_thres={prob_thres}\n")
            f.write(f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Contains results for {len(group_df)} WSIs\n\n")
            
            # Show T-score distribution for this threshold
            if 'tubulitis_score_ground_truth_numeric' in group_df.columns:
                t_scores = group_df['tubulitis_score_ground_truth_numeric'].round().astype(int).value_counts().sort_index()
                f.write("Ground truth T-score distribution:\n")
                for score, count in t_scores.items():
                    f.write(f"T{int(score)}: {count} samples\n")
        
        self.logger.info(f"Created comprehensive visualizations for threshold {prob_thres}")


    def _create_threshold_comparison_visualizations(self, metrics_df, eval_df, summary_dir, version_dir):
        """Create comparison visualizations across all probability thresholds."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import confusion_matrix
        
        # 1. Metrics comparison across thresholds
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['accuracy', 'f1_score', 'mae', 'kappa', 'pearson_corr', 'accuracy_within_one']
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes) and metric in metrics_df.columns:
                axes[i].plot(metrics_df['prob_thres'], metrics_df[metric], 'o-')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Probability Threshold')
                axes[i].grid(alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        comparison_path = summary_dir / "threshold_comparison.png"
        plt.savefig(comparison_path)
        plt.savefig(version_dir / "threshold_comparison.png")
        plt.close()
        
        # 2. Error by threshold (boxplot) - parameter effect on prediction error
        plt.figure(figsize=(8, 6))
        
        eval_df['error'] = np.abs(eval_df['tubulitis_score_predicted_numeric'] - eval_df['tubulitis_score_ground_truth_numeric'])
        sns.boxplot(data=eval_df, x='prob_thres', y='error')
        plt.title('Error by Probability Threshold')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Absolute Error')
        plt.grid(True, alpha=0.3)
        
        error_comparison_path = summary_dir / "parameter_effect_on_error.png"
        plt.savefig(error_comparison_path)
        plt.savefig(version_dir / "parameter_effect_on_error.png")
        plt.close()
        
        # 3. Find and visualize best parameter combinations
        if not metrics_df.empty:
            try:
                best_params = {
                    'accuracy': metrics_df.loc[metrics_df['accuracy'].idxmax()],
                    'f1_score': metrics_df.loc[metrics_df['f1_score'].idxmax()],
                    'mae': metrics_df.loc[metrics_df['mae'].idxmin()],
                    'kappa': metrics_df.loc[metrics_df['kappa'].idxmax()],
                    'weighted_accuracy': metrics_df.loc[metrics_df['weighted_accuracy'].idxmax()]
                }
                
                # Save best parameters to CSV
                best_params_df = pd.DataFrame(best_params).T
                best_params_path = summary_dir / "best_parameters.csv"
                best_params_df.to_csv(best_params_path)
                best_params_df.to_csv(version_dir / "best_parameters.csv")
                self.logger.info(f"Saved best parameters to {best_params_path}")
                
                # Create confusion matrix for best accuracy parameters
                if 'accuracy' in best_params:
                    best_acc_params = best_params['accuracy']
                    best_group = eval_df[(eval_df['prob_thres'] == best_acc_params['prob_thres'])]
                    
                    if len(best_group) > 0:
                        plt.figure(figsize=(8, 6))
                        
                        y_true = best_group['tubulitis_score_ground_truth_numeric'].round().astype(int)
                        y_pred = best_group['tubulitis_score_predicted_numeric'].round().astype(int)
                        
                        try:
                            # Check if we have enough unique classes for a meaningful confusion matrix
                            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
                            if len(unique_classes) > 1:
                                cm = confusion_matrix(y_true, y_pred)
                                
                                # Normalize confusion matrix
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                                    cm_norm = np.nan_to_num(cm_norm, nan=0)  # Replace NaNs with 0
                                
                                # Create heatmap
                                sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                                           xticklabels=[f't{i}' for i in range(4)],
                                           yticklabels=[f't{i}' for i in range(4)])
                                
                                plt.title(f'Confusion Matrix (Best Parameters: p={best_acc_params["prob_thres"]})')
                                plt.xlabel('Predicted T-score')
                                plt.ylabel('Ground Truth T-score')
                                
                                # Save plot
                                cm_path = summary_dir / "confusion_matrix.png"
                                plt.savefig(cm_path)
                                plt.savefig(version_dir / "confusion_matrix.png")
                            else:
                                self.logger.warning(f"Not enough unique classes for confusion matrix in best parameter group (p={best_acc_params['prob_thres']})")
                        except Exception as e:
                            self.logger.warning(f"Error creating confusion matrix: {e}")
                        plt.close()
                    
            except Exception as e:
                self.logger.warning(f"Error determining best parameters: {e}")
        
        # 4. Overall correlation between predicted and ground truth scores across all thresholds
        plt.figure(figsize=(10, 8))
        
        # Create a jittered scatter plot with different markers for different parameter combinations
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
        colors = plt.cm.tab10.colors
        
        param_groups = eval_df.groupby('prob_thres')
        for i, (prob_thres, group) in enumerate(param_groups):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            # Add jitter to avoid overplotting
            x = group['tubulitis_score_ground_truth_numeric'] + np.random.normal(0, 0.05, len(group))
            y = group['tubulitis_score_predicted_numeric'] + np.random.normal(0, 0.05, len(group))
            
            plt.scatter(x, y, marker=marker, color=color, alpha=0.7, 
                       label=f'p={prob_thres}')
        
        # Add perfect prediction line
        plt.plot([0, 3], [0, 3], 'k--', alpha=0.5)
        
        # Add regression line for all data
        x = eval_df['tubulitis_score_ground_truth_numeric']
        y = eval_df['tubulitis_score_predicted_numeric']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(x), p(np.sort(x)), "r--", alpha=0.8)
        
        # Calculate overall correlation
        try:
            if len(x) >= 2 and x.nunique() > 1 and y.nunique() > 1:
                from scipy.stats import pearsonr
                pearson_corr, _ = pearsonr(x, y)
                corr_text = f'Pearson r={pearson_corr:.2f}'
            else:
                self.logger.warning("Cannot calculate overall correlation: insufficient data variation")
                pearson_corr = np.nan
                corr_text = 'Correlation N/A'
        except Exception as e:
            self.logger.warning(f"Error calculating overall correlation: {e}")
            pearson_corr = np.nan
            corr_text = 'Correlation error'
        
        plt.title(f'Predicted vs Ground Truth T-scores ({corr_text})')
        plt.xlabel('Ground Truth T-score')
        plt.ylabel('Predicted T-score')
        plt.grid(alpha=0.3)
        plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        pred_vs_truth_path = summary_dir / "predicted_vs_ground_truth.png"
        plt.savefig(pred_vs_truth_path)
        plt.savefig(version_dir / "predicted_vs_ground_truth.png")
        plt.close()
        
        self.logger.info("Created threshold comparison visualizations")

    def run_by_stage(self, wsi_path: str, stage: str, force: bool = False, visualise: bool = False, update_summary: bool = False) -> Dict[str, Any]:
        # Handle comma-separated stages (e.g., "3,4")
        if "," in stage:
            stages = stage.split(",")
            self.logger.info(f"Running multiple stages: {stages}")
            results = {}
            
            for single_stage in stages:
                single_stage = single_stage.strip()
                stage_map = {"segment": "1", "detect": "2", "grade": "3"}
                # Convert named stages to numbers if needed
                if single_stage in stage_map:
                    single_stage = stage_map[single_stage]
                    
                self.logger.info(f"***** STARTING STAGE {single_stage} *****")
                stage_result = self.run_by_stage(
                    wsi_path, 
                    single_stage, 
                    force=force, 
                    visualise=visualise, 
                    update_summary=False  # Only update summary at the end
                )
                self.logger.info(f"***** COMPLETED STAGE {single_stage} *****")
                results[f"stage_{single_stage}"] = stage_result
            
            # Update summary at the end if requested
            if update_summary:
                self.create_summary_files(update_summary=True)
                
            return results
            
        # Handle individual stages
        if stage == "1":
            return self.run_stage1(wsi_path, force=force, visualise=visualise)
        elif stage == "2":
            return self.run_stage2(wsi_path, force=force, visualise=visualise)
        elif stage == "3":
            return self.run_stage3(wsi_path, force=force, visualise=visualise)
        elif stage == "all":
            return self.run_pipeline(wsi_path, force=force, visualise=visualise, update_summary=update_summary)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def run_pipeline(self, wsi_path: str, force: bool = False, visualise: bool = False, update_summary: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Starting full pipeline for {wsi_path}")
        self.run_stage1(wsi_path, force=force, visualise=visualise)
        self.logger.info("Stage 1 completed. Proceeding to Stage 2.")
        self.run_stage2(wsi_path, force=force, visualise=visualise)
        self.logger.info("Stage 2 completed. Proceeding to Stage 3.")
        result = self.run_stage3(wsi_path, force=force, visualise=visualise)
        
        # Create summary files after each completed pipeline run
        if update_summary:
            self.create_summary_files(update_summary=True)
        
        return result

def main():
    import argparse

    parser = argparse.ArgumentParser(description="KidneyGrader: the E2E pipeline for Banff scoring of kidney biopsies")
    
    parser.add_argument("--input_path", required=True, help="Full path to the WSI image or patch")
    parser.add_argument("--output_dir", default="results", help="Directory where results will be saved (default: results)")

    parser.add_argument("--stage", 
        default="all",
        help="Which stage(s) to run (default: all). Options: 1, 2, 3, all, segment, detect, grade. "
             "You can also specify multiple stages with commas, e.g., '2,3' for detection and grading."
    )

    parser.add_argument("--model_path", type=str, 
        default="checkpoints/segmentation/kidney_grader_unet.pth",
        help="Path to segmentation model checkpoint"
    )

    parser.add_argument("--force", action="store_true", help="Recompute all stages even if outputs exist")
    parser.add_argument("--visualise", action="store_true", help="Visualise segmentation results")

    parser.add_argument("--prob_thres", type=float, default=0.50,
                        help="Probability threshold (p ≥ value) used for inflammatory‑cell "
                        "filtering in stages 2 & 3 [default: 0.50]")
                        
    parser.add_argument("--update_summary", action="store_true",
                        help="Update summary files after processing")
                        
    parser.add_argument("--summary_only", action="store_true",
                        help="Only regenerate summary files without processing any WSIs")
                        
    parser.add_argument("--detection_json", type=str, default=None,
                        help="Path to a custom inflammatory cell detection JSON file to use instead of running detection")
                        
    parser.add_argument("--instance_mask_class1", type=str, default=None,
                        help="Path to a custom instance mask for class 1 (tubules) to use instead of running segmentation")

    args = parser.parse_args()

    console.print("[bold cyan]KidneyGrader Pipeline Starting...[/bold cyan]")
    console.print(f"[green]Input:[/green] {args.input_path}")
    console.print(f"[green]Stage:[/green] {args.stage}")
    console.print(f"[green]Output directory:[/green] {args.output_dir}")
    if args.detection_json:
        console.print(f"[green]Using custom detection JSON:[/green] {args.detection_json}")
    if args.instance_mask_class1:
        console.print(f"[green]Using custom instance mask:[/green]")
        console.print(f"  - Class 1 (tubules): {args.instance_mask_class1}")

    stage_map = {"segment": "1", "detect": "2", "grade": "3"}
    stage = stage_map.get(args.stage, args.stage)

    start = time.time()
    pipeline = KidneyGraderPipeline(
        output_dir=args.output_dir, 
        model_path=args.model_path, 
        prob_thres=args.prob_thres, 
        custom_detection_json=args.detection_json,
        custom_instance_mask_class1=args.instance_mask_class1
    )
    
    # Handle summary-only mode
    if args.summary_only:
        console.print("[bold yellow]Summary-only mode: Regenerating summary files...[/bold yellow]")
        pipeline.create_summary_files(update_summary=True)
        console.print("[bold green]Summary files updated.[/bold green]")
        return
    
    # Process the WSI
    result = pipeline.run_by_stage(
        args.input_path, 
        stage, 
        force=args.force, 
        visualise=args.visualise,
        update_summary=args.update_summary
    )
    
    # Additional summary updates for individual stage runs
    if stage != "all" and args.update_summary:
        pipeline.create_summary_files(update_summary=True)
        
    end = time.time()

    console.print(f"[bold green]Done in {end - start:.2f} seconds.[/bold green]")

    if stage in ("3", "grade"):
        console.print("[bold yellow]Grading complete.[/bold yellow]")
        console.print(f"WSI Name: {result['wsi_name']}")
        console.print(f"Tubulitis Grade: {result['tubulitis_score_predicted']}")
        console.print(f"Grading Report Path: {result['grading_report']}")
        console.print(
            f"Total tubules = {result['total_tubules']}, "
            f"total inflammatory cells = {result['total_inflam_cells']}, "
            f"mean cells/tubule = {result['summary_stats'].get('mean_cells_per_tubule', 0):.1f}"
        )
    else:
        console.print(f"[bold yellow]Result:[/bold yellow] {json.dumps(result, indent=2)}")
        
if __name__ == "__main__":
    main()
