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
from quantification.quantify import analyze_tubule_cell_distribution, convert_numpy_types, count_cells_in_tubules
from quantification.tubule_utils import identify_foci

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
    
    def __init__(self, output_dir: str, model_path: str = "checkpoints/best_current_model.pth", prob_thres: float = 0.80, foci_dist = 200, custom_detection_json: str = None, custom_instance_mask_class1: str = None, custom_instance_mask_class4: str = None):
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.prob_thres = prob_thres
        self.foci_dist = foci_dist
        self.custom_detection_json = custom_detection_json
        self.custom_instance_mask_class1 = custom_instance_mask_class1
        self.custom_instance_mask_class4 = custom_instance_mask_class4

        setup_logging(self.output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create main directory structure
        self.individual_reports_dir = self.output_dir / "individual_reports"
        self.summary_dir = self.output_dir / "summary"
        
        # Create main directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.individual_reports_dir.mkdir(exist_ok=True)
        self.summary_dir.mkdir(exist_ok=True)

    def get_output_paths(self, wsi_path: str) -> Dict[str, Path]:
        wsi_name = Path(wsi_path).stem
        
        # Create parameter-specific tag for directories
        prob_tag = f"p{self.prob_thres:.2f}".replace(".", "")
        dist_tag = f"d{self.foci_dist}"
        param_tag = f"{prob_tag}_{dist_tag}"
        
        # Create WSI-specific directory
        wsi_dir = self.individual_reports_dir / wsi_name
        wsi_dir.mkdir(exist_ok=True)
        
        # Create shared directories for segmentation and detection (param-independent)
        segmentation_dir = wsi_dir / "segmentation"
        detection_dir = wsi_dir / "detection"
        segmentation_dir.mkdir(exist_ok=True)
        detection_dir.mkdir(exist_ok=True)
        
        # Create parameter-specific directory for grading
        param_dir = wsi_dir / param_tag
        param_dir.mkdir(exist_ok=True)
        
        # Define quantification directory path but DON'T create it yet
        # Let the quantification stage create it only when needed
        quantification_dir = param_dir / "quantification"
        
        # Create parameters file
        params = {
            "prob_thres": self.prob_thres,
            "foci_dist": self.foci_dist,
            "model_path": str(self.model_path),
            "param_tag": param_tag
        }
        params_file = param_dir / "parameters.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)
        
        return {
            "wsi_name": wsi_name,
            "param_tag": param_tag,
            "tubule_mask": segmentation_dir / f"{wsi_name}_full_instance_mask_class1.tiff",
            "inflam_cell_mask": detection_dir / "detected-inflammatory-cells.json",
            "counts_csv": quantification_dir / f"{wsi_name}_tubule_counts.csv",
            "quant_json": quantification_dir / f"{wsi_name}_quantification.json",
            "grading_report": param_dir / "grading_report.json",
            "parameters": str(params_file)
        }
    
    def run_stage1(self, wsi_path: str, force: bool = False, visualise: bool = False) -> Dict[str, Any]:
        from segmentation.segment import run_segment

        self.logger.info(f"Running Stage 1: Segmentation for {wsi_path}")
        wsi_name = Path(wsi_path).stem
        paths = self.get_output_paths(wsi_path)
        
        # Use the shared segmentation directory
        output_dir = self.individual_reports_dir / wsi_name / "segmentation"
        semantic_mask_path = output_dir / f"{wsi_name}_semantic_mask.tiff"
        instance_mask_path1 = output_dir / f"{wsi_name}_full_instance_mask_class1.tiff"
        instance_mask_path4 = output_dir / f"{wsi_name}_full_instance_mask_class4.tiff"

        # Check if custom instance masks are provided
        custom_masks_used = False
        if self.custom_instance_mask_class1 and self.custom_instance_mask_class4:
            output_dir.mkdir(exist_ok=True)
            
            # Handle custom class 1 instance mask (tubules)
            custom_mask_path1 = Path(self.custom_instance_mask_class1)
            custom_mask_path4 = Path(self.custom_instance_mask_class4)
            
            # Check if both files exist
            if custom_mask_path1.exists() and custom_mask_path4.exists():
                self.logger.info(f"Using custom instance masks:")
                self.logger.info(f"  - Class 1 (tubules): {custom_mask_path1}")
                self.logger.info(f"  - Class 4 (glomeruli): {custom_mask_path4}")
                
                # Copy both files
                import shutil
                shutil.copy(custom_mask_path1, instance_mask_path1)
                shutil.copy(custom_mask_path4, instance_mask_path4)
                custom_masks_used = True
                
                # Return early with custom masks
                instance_mask_paths = {
                    1: str(instance_mask_path1),
                    4: str(instance_mask_path4)
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
                if not custom_mask_path4.exists():
                    missing_files.append(f"Class 4 mask: {custom_mask_path4}")
                    
                self.logger.warning(f"Custom instance masks not found: {', '.join(missing_files)}. Will run segmentation.")

        # check if both instance masks exist before skipping
        if instance_mask_path1.exists() and instance_mask_path4.exists() and not force:
            self.logger.info(f"Segmentation mask already exists at {output_dir}, skipping segmentation.")
            instance_mask_paths = {
                int(p.stem.split("class")[-1]): str(p)
                for p in output_dir.glob(f"{wsi_name}_full_instance_mask_class*.tiff")
            }
            return {
                "semantic_mask_path": str(semantic_mask_path),
                "instance_mask_paths": instance_mask_paths
            }

        # run segmentation if not cached
        result = run_segment(wsi_path, output_dir=output_dir, model_path=self.model_path, visualise=visualise)
        return result


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
            
            return {
                "wsi_name": wsi_name,
                "prob_threshold": self.prob_thres,
                "inflam_cell_coords_path": str(json_detection_path),
            }

        # If not cached, run detection
        output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Running inflammatory cell detection as the file does not exist or force flag is set.")
        run_inflammatory_cell_detection(
            wsi_path=wsi_path,
            output_dir=output_dir,
            model_path="detection/models/",
            visualise=visualise
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
        
        return {
            "wsi_name": wsi_name,
            "prob_threshold": self.prob_thres,
            "inflam_cell_coords_path": str(json_detection_path),
        }

    def run_stage3(self, wsi_path: str, force: bool = False, visualise: bool = False) -> dict:
        import tifffile as tiff
        self.logger.info("Running Stage 3: Quantification of inflammatory cells per tubule")

        paths = self.get_output_paths(wsi_path)
        wsi_name = Path(wsi_path).stem
        
        # Check if quantification already exists
        if paths["counts_csv"].exists() and paths["quant_json"].exists() and not force:
            # Load the existing results to check parameters
            try:
                with open(paths["quant_json"]) as f:
                    existing_results = json.load(f)
                
                # Check if parameters match
                existing_prob_thres = existing_results.get("prob_thres")
                existing_foci_dist = existing_results.get("foci_dist")
                
                if (existing_prob_thres == self.prob_thres and 
                    existing_foci_dist == self.foci_dist):
                    # Parameters match, we can reuse the existing results
                    self.logger.info(f"Quantification already exists for parameters prob_thres={self.prob_thres}, foci_dist={self.foci_dist}. Reusing results.")
                    
                    # If visualization is requested, generate it even with existing results
                    if visualise:
                        self.logger.info("Generating visualization for existing quantification results")
                        try:
                            from quantification.visualize_quantification import create_quantification_overlay
                            
                            # Load necessary data for visualization
                            tubule_mask_path = self.individual_reports_dir / wsi_name / "segmentation" / f"{wsi_name}_full_instance_mask_class1.tiff"
                            with tiff.TiffFile(tubule_mask_path) as tif:
                                tubule_mask = tif.asarray(out='memmap')
                                
                            # Load inflammatory cell detections
                            detection_json_path = self.individual_reports_dir / wsi_name / "detection" / "detected-inflammatory-cells.json"
                            with open(detection_json_path) as f:
                                inflam = json.load(f)
                            
                            # Load all points first
                            all_points = inflam["points"]
                            
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
                            
                            # Load counts from existing CSV
                            counts_df = pd.read_csv(paths["counts_csv"])
                            
                            # Create visualization directory
                            vis_dir = Path(paths["counts_csv"]).parent / "visualization"
                            
                            # Generate overlay
                            overlay_path = create_quantification_overlay(
                                wsi_path=wsi_path,
                                tubule_mask=tubule_mask,
                                cell_coords=mm_coords,
                                counts_df=counts_df,
                                output_dir=vis_dir
                            )
                            
                            if overlay_path:
                                self.logger.info(f"Created quantification visualization at {overlay_path}")
                                existing_results["visualization_path"] = str(overlay_path)
                            
                        except Exception as e:
                            self.logger.error(f"Failed to create visualization: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                    
                    return existing_results
                else:
                    # Parameters don't match, need to recompute
                    self.logger.info(f"Existing quantification found but parameters differ (existing: prob_thres={existing_prob_thres}, foci_dist={existing_foci_dist}, current: prob_thres={self.prob_thres}, foci_dist={self.foci_dist}). Recomputing.")
            except Exception as e:
                # If there's an error reading the file, recompute to be safe
                self.logger.warning(f"Error reading existing quantification: {e}. Recomputing.")
        
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
            tubule_mask = tif.asarray(out='memmap')  # Lazy load to avoid memory overload

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
        cell_coords = (mm_coords * 1000 / MICRONS_PER_PIXEL).astype(np.int32)

        from tiffslide import TiffSlide
        slide = TiffSlide(wsi_path)
        self.logger.info(f"[Stage 3] WSI name: {paths['wsi_name']}")
        self.logger.info(f"[Stage 3] WSI size: {slide.dimensions}")
        self.logger.info(f"[Stage 3] Tubule mask shape: {tubule_mask.shape}")
        self.logger.info(f"[Stage 3] Parameters: prob_thres={self.prob_thres}, foci_dist={self.foci_dist}")

        # Generate foci with parameter-specific distance
        foci_mask = identify_foci(tubule_mask, min_distance=self.foci_dist)
        counts_df = count_cells_in_tubules(cell_coords, tubule_mask, foci_mask)

        # Create quantification directory only when we're actually saving results
        quantification_dir = Path(paths["counts_csv"]).parent
        quantification_dir.mkdir(exist_ok=True)
        
        # Save results to parameter-specific directory and get summary stats in one step
        # (avoiding duplicate call to analyze_tubule_cell_distribution)
        self.logger.info("Saving counts and computing summary statistics...")
        
        # First, save the CSV and get summary stats (without redundant analysis)
        counts_df.to_csv(paths["counts_csv"], index=False)
        summary_stats = analyze_tubule_cell_distribution(counts_df)
        
        # Save summary stats separately
        summary_path = Path(paths["counts_csv"]).parent / "summary_stats.csv"
        pd.DataFrame([summary_stats]).to_csv(summary_path, index=False)
        
        self.logger.info(f"Saved cell counts to {paths['counts_csv']}")
        self.logger.info(f"Saved summary statistics to {summary_path}")
        
        # Create visualization if requested
        visualization_path = None
        if visualise:
            self.logger.info("Generating visualization for quantification results")
            try:
                from quantification.visualize_quantification import create_quantification_overlay
                
                # Create visualization directory
                vis_dir = quantification_dir / "visualization"
                
                # Generate overlay
                visualization_path = create_quantification_overlay(
                    wsi_path=wsi_path,
                    tubule_mask=tubule_mask,
                    cell_coords=mm_coords,
                    counts_df=counts_df,
                    output_dir=vis_dir
                )
                
                if visualization_path:
                    self.logger.info(f"Created quantification visualization at {visualization_path}")
            except Exception as e:
                self.logger.error(f"Failed to create visualization: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Create per_tubule_counts directly in the format needed for output
        # (avoiding duplicate dictionary creation)
        per_tubule_counts = {int(k): int(v) for k, v in zip(counts_df['tubule_id'], counts_df['cell_count'])}

        output = {
            "wsi_name": paths["wsi_name"],
            "total_inflam_cells": int(len(cell_coords)),
            "total_tubules": int(len(np.unique(tubule_mask)) - 1),
            "tubule_counts_csv": str(paths["counts_csv"]),
            "per_tubule_inflam_cell_counts": per_tubule_counts,  # Already in the correct format
            "summary_stats": convert_numpy_types(summary_stats),
            "prob_thres": self.prob_thres,
            "foci_dist": self.foci_dist,
            "param_tag": paths["param_tag"]
        }
        
        # Add visualization path if available
        if visualization_path:
            output["visualization_path"] = str(visualization_path)

        with open(paths["quant_json"], "w") as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Saved structured summary to {paths['quant_json']}")
        self.logger.info("====== STAGE 3: QUANTIFICATION COMPLETED ======")  # Add explicit stage marker
        return output

    def create_summary_files(self, update_summary: bool = True) -> None:
        """Create or update summary files with all results across parameter sets.
        
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
        
        # Collect all grading results across parameter sets
        results = []
        
        for wsi_dir in self.individual_reports_dir.iterdir():
            if not wsi_dir.is_dir():
                continue
                
            wsi_name = wsi_dir.name
            
            # Look through parameter-specific directories
            for param_dir in wsi_dir.iterdir():
                if not param_dir.is_dir() or not param_dir.name.startswith("p"):
                    continue
                    
                grading_report_path = param_dir / "grading_report.json"
                summary_stats_path = param_dir / "summary_stats.csv"
                
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
                        "foci_dist": data.get("foci_dist", None),
                        "param_tag": data.get("param_tag", param_dir.name),
                        "total_inflammatory_tubules": data.get("Total inflammatory tubules", 0),
                        "total_foci": data.get("Total foci", 0),
                        "max_cells_in_tubule": data.get("Max cells in any tubule", 0)
                    }
                    
                    # Add additional metrics from summary_stats if available
                    if summary_stats_path.exists():
                        try:
                            summary_df = pd.read_csv(summary_stats_path)
                            if not summary_df.empty:
                                result.update({
                                    "total_tubules": summary_df["total_tubules"].iloc[0],
                                    "total_cells": summary_df["total_cells"].iloc[0],
                                    "mean_cells_per_tubule": summary_df["mean_cells_per_tubule"].iloc[0],
                                    "std_cells_per_tubule": summary_df["std_cells_per_tubule"].iloc[0]
                                })
                                
                                # Try to extract focus stats if available
                                if "focus_stats" in summary_df.columns:
                                    try:
                                        focus_stats = eval(summary_df["focus_stats"].iloc[0])
                                        if focus_stats and len(focus_stats) > 0:
                                            # Calculate average metrics across foci
                                            avg_tubules_per_focus = np.mean([f.get('num_tubules', 0) for f in focus_stats if f.get('focus_id', 0) > 0])
                                            avg_cells_per_focus = np.mean([f.get('total_cells', 0) for f in focus_stats if f.get('focus_id', 0) > 0])
                                            avg_cells_per_tubule_in_foci = np.mean([f.get('mean_cells_per_tubule', 0) for f in focus_stats if f.get('focus_id', 0) > 0])
                                            max_cells_in_any_focus = max([f.get('max_cells_in_tubule', 0) for f in focus_stats if f.get('focus_id', 0) > 0], default=0)
                                            
                                            result.update({
                                                "avg_tubules_per_focus": avg_tubules_per_focus,
                                                "avg_cells_per_focus": avg_cells_per_focus,
                                                "avg_cells_per_tubule_in_foci": avg_cells_per_tubule_in_foci,
                                                "max_cells_in_any_focus": max_cells_in_any_focus
                                            })
                                    except Exception as e:
                                        self.logger.warning(f"Could not parse focus stats for {wsi_name}: {e}")
                        except Exception as e:
                            self.logger.warning(f"Could not read summary stats for {wsi_name}: {e}")
                    
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
        
        # Create directory for summary files
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the raw data
        scores_csv = self.summary_dir / "aggregated_scores.csv"
        results_df.to_csv(scores_csv, index=False)
        self.logger.info(f"Saved aggregated scores to {scores_csv}")
        
        # Set up versioned directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.summary_dir / f"version_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(version_dir / "aggregated_scores.csv", index=False)
        
        # Filter to only include results with ground truth for evaluation
        eval_df = results_df.dropna(subset=['tubulitis_score_ground_truth_numeric', 'tubulitis_score_predicted_numeric'])
        
        if len(eval_df) == 0:
            self.logger.warning("No results with ground truth available for evaluation")
            return
            
        self.logger.info(f"Evaluating {len(eval_df)} results with ground truth data")
        
        # Calculate evaluation metrics by parameter combination
        param_groups = eval_df.groupby(['prob_thres', 'foci_dist'])
        
        # Prepare metrics dataframe
        metrics_data = []
        
        for (prob_thres, foci_dist), group in param_groups:
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
                    self.logger.warning(f"Could not calculate correlation for group (p={prob_thres}, d={foci_dist}): {e}")
                    pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
            else:
                self.logger.warning(f"Group (p={prob_thres}, d={foci_dist}) has only {len(group)} samples - skipping correlation calculation")
                pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
            
            # Calculate Cohen's Kappa (agreement metric)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # Calculate custom metrics
            # Accuracy within 1 score point
            within_one = np.mean(np.abs(group['tubulitis_score_ground_truth_numeric'] - group['tubulitis_score_predicted_numeric']) <= 1)
            
            # Weighted accuracy (penalizes larger errors more)
            weights = 1.0 / (1.0 + np.abs(group['tubulitis_score_ground_truth_numeric'] - group['tubulitis_score_predicted_numeric']))
            weighted_accuracy = np.mean(weights)
            
            metrics_data.append({
                'prob_thres': prob_thres,
                'foci_dist': foci_dist,
                'sample_count': len(group),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'kappa': kappa,
                'accuracy_within_one': within_one,
                'weighted_accuracy': weighted_accuracy
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Save metrics to CSV
        metrics_path = self.summary_dir / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        metrics_df.to_csv(version_dir / "evaluation_metrics.csv", index=False)
        self.logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        # Find best parameter combinations based on different metrics
        best_params = {}
        
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
                best_params_path = self.summary_dir / "best_parameters.csv"
                best_params_df.to_csv(best_params_path)
                best_params_df.to_csv(version_dir / "best_parameters.csv")
                self.logger.info(f"Saved best parameters to {best_params_path}")
            except Exception as e:
                self.logger.warning(f"Error determining best parameters: {e}")
        else:
            self.logger.warning("No metrics data available - skipping best parameters calculation")
        
        # Create correlation analysis between cell counts and ground truth scores
        if len(eval_df) > 0:
            correlation_data = []
            
            # Features to correlate with ground truth scores
            features = [
                'total_inflammatory_tubules', 'total_foci', 'max_cells_in_tubule',
                'mean_cells_per_tubule', 'total_cells', 'avg_cells_per_focus',
                'max_cells_in_any_focus'
            ]
            
            # Calculate correlations for each parameter combination
            for (prob_thres, foci_dist), group in param_groups:
                for feature in features:
                    if feature in group.columns:
                        # Calculate correlations
                        try:
                            # Skip if there are fewer than 2 samples or all values are identical
                            if len(group) < 2 or group[feature].nunique() <= 1 or group['tubulitis_score_ground_truth_numeric'].nunique() <= 1:
                                self.logger.warning(f"Skipping correlation for feature {feature} in group (p={prob_thres}, d={foci_dist}): insufficient data variation")
                                continue
                                
                            pearson_corr, pearson_p = pearsonr(
                                group[feature], 
                                group['tubulitis_score_ground_truth_numeric']
                            )
                            spearman_corr, spearman_p = spearmanr(
                                group[feature], 
                                group['tubulitis_score_ground_truth_numeric']
                            )
                            
                            correlation_data.append({
                                'prob_thres': prob_thres,
                                'foci_dist': foci_dist,
                                'feature': feature,
                                'pearson_corr': pearson_corr,
                                'pearson_p': pearson_p,
                                'spearman_corr': spearman_corr,
                                'spearman_p': spearman_p
                            })
                        except Exception as e:
                            self.logger.warning(f"Could not calculate correlation for {feature} in group (p={prob_thres}, d={foci_dist}): {e}")
            
            if correlation_data:
                correlation_df = pd.DataFrame(correlation_data)
                correlation_path = self.summary_dir / "feature_correlations.csv"
                correlation_df.to_csv(correlation_path, index=False)
                correlation_df.to_csv(version_dir / "feature_correlations.csv", index=False)
                self.logger.info(f"Saved feature correlations to {correlation_path}")
        
        # Create visualizations
        
        # 1. Correlation between predicted and ground truth scores
        plt.figure(figsize=(10, 8))
        
        # Create a jittered scatter plot with different markers for different parameter combinations
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
        colors = plt.cm.tab10.colors
        
        for i, ((prob_thres, foci_dist), group) in enumerate(param_groups):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            # Add jitter to avoid overplotting
            x = group['tubulitis_score_ground_truth_numeric'] + np.random.normal(0, 0.05, len(group))
            y = group['tubulitis_score_predicted_numeric'] + np.random.normal(0, 0.05, len(group))
            
            plt.scatter(x, y, marker=marker, color=color, alpha=0.7, 
                       label=f'p={prob_thres}, d={foci_dist}')
        
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
        pred_vs_truth_path = self.summary_dir / "predicted_vs_ground_truth.png"
        plt.savefig(pred_vs_truth_path)
        plt.savefig(version_dir / "predicted_vs_ground_truth.png")
        plt.close()
        
        # 2. Feature importance plot - correlation of features with ground truth
        if 'correlation_df' in locals():
            plt.figure(figsize=(12, 8))
            
            # Filter to significant correlations
            sig_correlations = correlation_df[correlation_df['pearson_p'] < 0.05]
            
            if not sig_correlations.empty:
                try:
                    # Pivot to get features as rows and parameters as columns
                    pivot_df = sig_correlations.pivot_table(
                        index='feature',
                        columns=['prob_thres', 'foci_dist'],
                        values='pearson_corr'
                    )
                    
                    if not pivot_df.empty:
                        # Plot heatmap
                        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                        plt.title('Feature Correlation with Ground Truth T-score')
                        plt.tight_layout()
                        
                        # Save plot
                        feature_corr_path = self.summary_dir / "feature_correlations.png"
                        plt.savefig(feature_corr_path)
                        plt.savefig(version_dir / "feature_correlations.png")
                    else:
                        self.logger.warning("Feature correlation pivot table is empty - skipping heatmap")
                except Exception as e:
                    self.logger.warning(f"Error creating feature correlation heatmap: {e}")
                plt.close()
            else:
                self.logger.warning("No significant correlations found - skipping feature correlation heatmap")
                plt.close()
        
        # 3. Parameter effect on prediction error
        plt.figure(figsize=(12, 6))
        
        # Calculate error for each sample
        eval_df['error'] = np.abs(eval_df['tubulitis_score_predicted_numeric'] - eval_df['tubulitis_score_ground_truth_numeric'])
        
        # Plot error by probability threshold
        plt.subplot(1, 2, 1)
        sns.boxplot(data=eval_df, x='prob_thres', y='error')
        plt.title('Error by Probability Threshold')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Absolute Error')
        plt.grid(True, alpha=0.3)
        
        # Plot error by foci distance
        plt.subplot(1, 2, 2)
        sns.boxplot(data=eval_df, x='foci_dist', y='error')
        plt.title('Error by Foci Distance')
        plt.xlabel('Foci Distance')
        plt.ylabel('Absolute Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        param_effect_path = self.summary_dir / "parameter_effect_on_error.png"
        plt.savefig(param_effect_path)
        plt.savefig(version_dir / "parameter_effect_on_error.png")
        plt.close()
        
        # 4. Confusion matrix for best parameters
        if best_params and 'accuracy' in best_params:
            best_acc_params = best_params['accuracy']
            best_group = eval_df[(eval_df['prob_thres'] == best_acc_params['prob_thres']) & 
                                (eval_df['foci_dist'] == best_acc_params['foci_dist'])]
            
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
                        
                        plt.title(f'Confusion Matrix (Best Parameters: p={best_acc_params["prob_thres"]}, d={best_acc_params["foci_dist"]})')
                        plt.xlabel('Predicted T-score')
                        plt.ylabel('Ground Truth T-score')
                        
                        # Save plot
                        cm_path = self.summary_dir / "confusion_matrix.png"
                        plt.savefig(cm_path)
                        plt.savefig(version_dir / "confusion_matrix.png")
                    else:
                        self.logger.warning(f"Not enough unique classes for confusion matrix in best parameter group (p={best_acc_params['prob_thres']}, d={best_acc_params['foci_dist']})")
                except Exception as e:
                    self.logger.warning(f"Error creating confusion matrix: {e}")
                plt.close()
        else:
            self.logger.warning("No best parameters available - skipping confusion matrix")
        
        # 5. Feature correlation with prediction error
        plt.figure(figsize=(10, 8))
        
        # Calculate correlations between features and error
        error_corr_data = []
        
        for feature in features:
            if feature in eval_df.columns:
                try:
                    # Skip if there are fewer than 2 samples or all values are identical
                    if len(eval_df) < 2 or eval_df[feature].nunique() <= 1 or eval_df['error'].nunique() <= 1:
                        self.logger.warning(f"Skipping error correlation for feature {feature}: insufficient data variation")
                        continue
                        
                    pearson_corr, pearson_p = pearsonr(eval_df[feature], eval_df['error'])
                    error_corr_data.append({
                        'feature': feature,
                        'correlation': pearson_corr,
                        'p_value': pearson_p
                    })
                except Exception as e:
                    self.logger.warning(f"Could not calculate error correlation for {feature}: {e}")
        
        if error_corr_data:
            try:
                error_corr_df = pd.DataFrame(error_corr_data)
                error_corr_df = error_corr_df.sort_values('correlation', ascending=False)
                
                # Create bar plot
                plt.barh(error_corr_df['feature'], error_corr_df['correlation'])
                plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                plt.title('Feature Correlation with Prediction Error')
                plt.xlabel('Pearson Correlation')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                error_corr_path = self.summary_dir / "error_correlations.png"
                plt.savefig(error_corr_path)
                plt.savefig(version_dir / "error_correlations.png")
                
                # Save to CSV
                error_corr_df.to_csv(self.summary_dir / "error_correlations.csv", index=False)
                error_corr_df.to_csv(version_dir / "error_correlations.csv", index=False)
            except Exception as e:
                self.logger.warning(f"Error creating error correlation visualization: {e}")
        else:
            self.logger.warning("No valid error correlations found - skipping error correlation visualization")
        
        plt.close()
        
        # 6. Specific analysis of mean inflammatory cells per tubule vs tubulitis score
        if 'mean_cells_per_tubule' in eval_df.columns:
            plt.figure(figsize=(12, 10))
            
            # Create scatter plot with regression line
            plt.subplot(2, 1, 1)
            sns.regplot(
                data=eval_df, 
                x='mean_cells_per_tubule', 
                y='tubulitis_score_ground_truth_numeric',
                scatter_kws={'alpha': 0.7},
                line_kws={'color': 'red'}
            )
            
            # Calculate correlation
            try:
                if len(eval_df) >= 2 and eval_df['mean_cells_per_tubule'].nunique() > 1:
                    pearson_r, p_value = pearsonr(
                        eval_df['mean_cells_per_tubule'], 
                        eval_df['tubulitis_score_ground_truth_numeric']
                    )
                    spearman_r, spearman_p = spearmanr(
                        eval_df['mean_cells_per_tubule'], 
                        eval_df['tubulitis_score_ground_truth_numeric']
                    )
                    
                    plt.title(f'Mean Cells per Tubule vs Ground Truth T-score\n'
                             f'Pearson r={pearson_r:.3f} (p={p_value:.4f}), '
                             f'Spearman r={spearman_r:.3f} (p={spearman_p:.4f})')
                else:
                    plt.title('Mean Cells per Tubule vs Ground Truth T-score\n'
                             'Not enough data variation for correlation calculation')
            except Exception as e:
                self.logger.warning(f"Error calculating correlation for mean_cells_per_tubule: {e}")
                plt.title('Mean Cells per Tubule vs Ground Truth T-score')
            
            plt.xlabel('Mean Inflammatory Cells per Tubule')
            plt.ylabel('Ground Truth T-score')
            plt.grid(True, alpha=0.3)
            
            # Create box plot to show distribution by T-score
            plt.subplot(2, 1, 2)
            
            # Ensure T-score is formatted correctly for the box plot
            eval_df['t_score_category'] = eval_df['tubulitis_score_ground_truth_numeric'].round().astype(int).map(lambda x: f't{x}')
            
            # Create box plot
            sns.boxplot(data=eval_df, x='t_score_category', y='mean_cells_per_tubule')
            plt.title('Distribution of Mean Cells per Tubule by T-score')
            plt.xlabel('T-score')
            plt.ylabel('Mean Inflammatory Cells per Tubule')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plots
            cells_vs_tscore_path = self.summary_dir / "mean_cells_vs_tscore.png"
            plt.savefig(cells_vs_tscore_path)
            plt.savefig(version_dir / "mean_cells_vs_tscore.png")
            
            # Save the data points for this specific relationship
            cells_vs_tscore_data = eval_df[['wsi_name', 'mean_cells_per_tubule', 
                                           'tubulitis_score_ground_truth', 
                                           'tubulitis_score_ground_truth_numeric',
                                           'tubulitis_score_predicted',
                                           'prob_thres', 'foci_dist']]
            cells_vs_tscore_path_csv = self.summary_dir / "mean_cells_vs_tscore.csv"
            cells_vs_tscore_data.to_csv(cells_vs_tscore_path_csv, index=False)
            cells_vs_tscore_data.to_csv(version_dir / "mean_cells_vs_tscore.csv", index=False)
            
            self.logger.info(f"Saved mean cells per tubule vs T-score analysis to {cells_vs_tscore_path}")
        else:
            self.logger.warning("No 'mean_cells_per_tubule' feature available - skipping specific analysis")
        
        plt.close()
        
        # Create a README for the version directory
        with open(version_dir / "README.txt", "w") as f:
            f.write(f"Summary analysis created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Contains results for {len(results_df)} WSIs across {len(param_groups)} parameter combinations\n")
            f.write(f"Evaluation performed on {len(eval_df)} WSIs with ground truth data\n\n")
            
            if best_params:
                f.write("Best parameter combinations:\n")
                for metric, params in best_params.items():
                    f.write(f"- Best for {metric}: prob_thres={params['prob_thres']}, foci_dist={params['foci_dist']} ")
                    f.write(f"(value: {params[metric]:.4f}, samples: {params['sample_count']})\n")
            else:
                f.write("No best parameter combinations determined due to insufficient data\n")
            
            # Show T-score distribution
            if 'tubulitis_score_ground_truth_numeric' in eval_df.columns:
                t_scores = eval_df['tubulitis_score_ground_truth_numeric'].round().astype(int).value_counts().sort_index()
                f.write("\nGround truth T-score distribution:\n")
                for score, count in t_scores.items():
                    f.write(f"T{int(score)}: {count} samples\n")
        
        self.logger.info(f"Updated summary files in {self.summary_dir}")
        self.logger.info(f"Created versioned summary in {version_dir}")

    def run_stage4(self, wsi_path: str, force: bool = False) -> dict:
        import pandas as pd

        self.logger.info("Running Stage 4: Grading")
        self.logger.info("====== STAGE 4: GRADING STAGE STARTED ======")  # Add explicit stage marker

        paths = self.get_output_paths(wsi_path)
        param_tag = paths["param_tag"]

        if paths["grading_report"].exists() and not force:
            # Load existing grading report to check parameters
            try:
                with open(paths["grading_report"]) as f:
                    grading_result = json.load(f)
                
                # Check if parameters match
                existing_prob_thres = grading_result.get("prob_thres")
                existing_foci_dist = grading_result.get("foci_dist")
                
                if (existing_prob_thres == self.prob_thres and 
                    existing_foci_dist == self.foci_dist):
                    # Parameters match, we can reuse the existing results
                    self.logger.info(f"Grading already exists for parameters {param_tag}. Reusing results.")
                    return grading_result
                else:
                    # Parameters don't match, need to recompute
                    self.logger.info(f"Existing grading found but parameters differ (existing: prob_thres={existing_prob_thres}, foci_dist={existing_foci_dist}, current: prob_thres={self.prob_thres}, foci_dist={self.foci_dist}). Recomputing.")
            except Exception as e:
                # If there's an error reading the file, recompute to be safe
                self.logger.warning(f"Error reading existing grading: {e}. Recomputing.")

        if not paths["counts_csv"].exists():
            raise FileNotFoundError(f"Required quantification CSV not found at {paths['counts_csv']}. Run Stage 3 first.")

        # Load the per-tubule DataFrame
        counts_df = pd.read_csv(paths["counts_csv"])

        # Calculate score using loaded counts
        grading_result = calculate_tubulitis_score(
            counts_df=counts_df,
            output_dir=paths["grading_report"].parent
        )

        # Check if we have ground truth T score
        wsi_name = Path(wsi_path).stem
        true_t_score = None
        
        # Try to get ground truth from banff_scores.csv if it exists
        banff_csv = Path("banff_scores.csv")
        if banff_csv.exists():
            try:
                import pandas as pd
                banff_df = pd.read_csv(banff_csv)
                
                # Try to find this WSI in the ground truth data
                wsi_filename = f"{wsi_name}.svs"
                match = banff_df[banff_df["filename"] == wsi_filename]
                
                if not match.empty and "T" in match.columns and pd.notna(match["T"].values[0]):
                    true_t_score = float(match["T"].values[0])
                    self.logger.info(f"Found ground truth T score: {true_t_score} for {wsi_name}")
            except Exception as e:
                self.logger.warning(f"Could not load ground truth T score: {e}")

        # Add parameter information to grading result
        final_result = {
            "wsi_name": paths["wsi_name"],
            "tubulitis_score_predicted": grading_result["score"],
            "grading_report": str(paths["grading_report"]),
            "prob_thres": self.prob_thres,
            "foci_dist": self.foci_dist,
            "param_tag": param_tag
        }
        
        # Add ground truth if available
        if true_t_score is not None:
            # Store ground truth in the same format as prediction (t0, t1, t2, t3)
            final_result["tubulitis_score_ground_truth"] = f"t{int(round(true_t_score))}"
            
            # Extract numeric value from prediction for comparison
            pred_score = float(grading_result["score"][1:]) if grading_result["score"].startswith("t") else float(grading_result["score"])
            
            # Calculate difference and correctness
            final_result["score_difference"] = abs(pred_score - true_t_score)
            final_result["correct_category"] = (round(pred_score) == round(true_t_score))
            
        # Ensure consistent ordering with tubulitis_score_ground_truth directly after tubulitis_score_predicted
        ordered_keys = ["wsi_name", "tubulitis_score_predicted", "tubulitis_score_ground_truth", 
                       "score_difference", "correct_category", "grading_report", 
                       "prob_thres", "foci_dist", "param_tag"]
        
        # Create a new ordered dictionary with the desired key order
        ordered_result = {}
        for key in ordered_keys:
            if key in final_result:
                ordered_result[key] = final_result[key]
                
        # Add any remaining keys not in the ordered list
        for key in final_result:
            if key not in ordered_result:
                ordered_result[key] = final_result[key]
                
        # Save the enhanced result with ordered keys
        with open(paths["grading_report"], "w") as f:
            json.dump(ordered_result, f, indent=2)

        self.logger.info(f"Grading report saved to {paths['grading_report']}")
        return final_result
        
    def run_by_stage(self, wsi_path: str, stage: str, force: bool = False, visualise: bool = False, update_summary: bool = False) -> Dict[str, Any]:
        # Handle comma-separated stages (e.g., "3,4")
        if "," in stage:
            stages = stage.split(",")
            self.logger.info(f"Running multiple stages: {stages}")
            results = {}
            
            for single_stage in stages:
                single_stage = single_stage.strip()
                stage_map = {"segment": "1", "detect": "2", "quantify": "3", "grade": "4"}
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
        elif stage == "4":
            return self.run_stage4(wsi_path, force=force)
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
        self.run_stage3(wsi_path, force=force, visualise=visualise)
        self.logger.info("Stage 3 completed. Proceeding to Stage 4.")
        result = self.run_stage4(wsi_path, force=force)
        
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
        help="Which stage(s) to run (default: all). Options: 1, 2, 3, 4, all, segment, detect, quantify, grade. "
             "You can also specify multiple stages with commas, e.g., '3,4' for quantification and grading."
    )

    parser.add_argument("--model_path", type=str, 
        default="checkpoints/best_current_model.pth",
        help="Path to segmentation model checkpoint"
    )

    parser.add_argument("--force", action="store_true", help="Recompute all stages even if outputs exist")
    parser.add_argument("--visualise", action="store_true", help="Visualise segmentation results")

    parser.add_argument("--prob_thres", type=float, default=0.80,
                        help="Probability threshold (p ≥ value) used for inflammatory‑cell "
                        "filtering in stages 2 & 3 [default: 0.80]")
    
    parser.add_argument("--foci_dist", type=int, default=200,
                        help="Minimum distance between foci in pixels [default: 200]")
                        
    parser.add_argument("--update_summary", action="store_true",
                        help="Update summary files after processing")
                        
    parser.add_argument("--summary_only", action="store_true",
                        help="Only regenerate summary files without processing any WSIs")
                        
    parser.add_argument("--detection_json", type=str, default=None,
                        help="Path to a custom inflammatory cell detection JSON file to use instead of running detection")
                        
    parser.add_argument("--instance_mask_class1", type=str, default=None,
                        help="Path to a custom instance mask for class 1 (tubules) to use instead of running segmentation")
                        
    parser.add_argument("--instance_mask_class4", type=str, default=None,
                        help="Path to a custom instance mask for class 4 (glomeruli) to use instead of running segmentation")

    args = parser.parse_args()

    # Validate that both instance masks are provided if either one is specified
    if (args.instance_mask_class1 and not args.instance_mask_class4) or (args.instance_mask_class4 and not args.instance_mask_class1):
        parser.error("Both --instance_mask_class1 and --instance_mask_class4 must be provided together")

    console.print("[bold cyan]KidneyGrader Pipeline Starting...[/bold cyan]")
    console.print(f"[green]Input:[/green] {args.input_path}")
    console.print(f"[green]Stage:[/green] {args.stage}")
    console.print(f"[green]Output directory:[/green] {args.output_dir}")
    if args.detection_json:
        console.print(f"[green]Using custom detection JSON:[/green] {args.detection_json}")
    if args.instance_mask_class1 and args.instance_mask_class4:
        console.print(f"[green]Using custom instance masks:[/green]")
        console.print(f"  - Class 1 (tubules): {args.instance_mask_class1}")
        console.print(f"  - Class 4 (glomeruli): {args.instance_mask_class4}")

    stage_map = {"segment": "1", "detect": "2", "quantify": "3", "grade": "4"}
    stage = stage_map.get(args.stage, args.stage)

    start = time.time()
    pipeline = KidneyGraderPipeline(
        output_dir=args.output_dir, 
        model_path=args.model_path, 
        prob_thres=args.prob_thres, 
        foci_dist=args.foci_dist,
        custom_detection_json=args.detection_json,
        custom_instance_mask_class1=args.instance_mask_class1,
        custom_instance_mask_class4=args.instance_mask_class4
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

    if stage in ("3", "quantify"):
        console.print("[bold yellow]Quantification complete.[/bold yellow]")
        console.print(
            f"Total tubules = {result['total_tubules']}, "
            f"total inflammatory cells = {result['total_inflam_cells']}, "
            f"mean cells/tubule = {result['summary_stats']['mean_cells_per_tubule']:.1f}"
        )
    elif stage in ("4", "grade"):
        console.print(f"[bold yellow]Grading complete.[/bold yellow]")
        console.print(f"WSI Name: {result['wsi_name']}")
        console.print(f"Tubulitis Grade: {result['tubulitis_score_predicted']}")
        console.print(f"Grading Report Path: {result['grading_report']}")
    else:
        console.print(f"[bold yellow]Result:[/bold yellow] {json.dumps(result, indent=2)}")
        
if __name__ == "__main__":
    main()
