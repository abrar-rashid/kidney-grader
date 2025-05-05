import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import time
import cv2
from rich.console import Console
from rich.progress import Progress
from grading.banff_grade import calculate_tubulitis_score
from quantification.quantify import analyze_tubule_cell_distribution, convert_numpy_types, count_cells_in_tubules, save_counts_csv
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
    
    def __init__(self, output_dir: str, model_path: str = "checkpoints/improved_unet_best.pth"): # TODO add instanseg model path here
        self.output_dir = Path(output_dir)
        self.model_path = model_path

        setup_logging(self.output_dir)
        self.logger = logging.getLogger(__name__)
        
        # output dirs are stage-specific
        self.segmentation_dir = self.output_dir / "segmentation"
        self.detection_dir = self.output_dir / "detection"
        self.quantification_dir = self.output_dir / "quantification"
        self.grading_dir = self.output_dir / "grading"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segmentation_dir.mkdir(exist_ok=True)
        self.detection_dir.mkdir(exist_ok=True)
        self.quantification_dir.mkdir(exist_ok=True)
        self.grading_dir.mkdir(exist_ok=True)

    def get_output_paths(self, wsi_path: str) -> Dict[str, Path]:
        wsi_name = Path(wsi_path).stem
        return {
            "wsi_name": wsi_name,
            "tubule_mask": self.segmentation_dir / wsi_name / f"{wsi_name}_full_instance_mask_class1.npy",
            "inflam_cell_mask": self.detection_dir / wsi_name / "inflam_cell_instance_mask.npy",
            "counts_csv": self.quantification_dir / f"{wsi_name}_tubule_counts.csv",
            "quant_json": self.quantification_dir / f"{wsi_name}_quantification.json",
            "grading_report": self.grading_dir / f"{wsi_name}_grading_report.json"
        }
    
    def run_stage1(self, wsi_path: str, force: bool = False, visualise: bool = False) -> Dict[str, Any]:
        from segmentation.segment import run_segment
        # to run semantic segmentation of WSI

        self.logger.info(f"Running Stage 1: Segmentation for {wsi_path}")
        wsi_name = Path(wsi_path).stem
        output_dir = self.segmentation_dir / wsi_name
        semantic_mask_npy_path = output_dir / f"{wsi_name}_semantic_mask.npy"
        semantic_mask_png_path = output_dir / f"{wsi_name}_semantic_mask.png"

        if semantic_mask_npy_path.exists() and not force:
            self.logger.info(f"Segmentation mask already exists at {semantic_mask_npy_path}, skipping segmentation.")
            instance_mask_paths = {
                int(p.stem.split("class")[-1]): str(p)
                for p in output_dir.glob(f"{wsi_name}_full_instance_mask_class*.npy")
            }
            return {
                "semantic_mask_png_path": str(semantic_mask_png_path),
                "semantic_mask_npy_path": str(semantic_mask_npy_path),
                "instance_mask_paths": instance_mask_paths
            }

        result = run_segment(wsi_path, output_dir=self.segmentation_dir, model_path=self.model_path, visualise=visualise)
        return result

    def run_stage2(self, wsi_path, force: bool = False, visualise: bool = False) -> dict:
        from detection.detect import run_inflammatory_cell_detection

        self.logger.info("Running Stage 2: Inflammatory cell detection")
        wsi_name = Path(wsi_path).stem
        output_dir = self.detection_dir / wsi_name
        mm_coords_path = output_dir / "inflam_cell_mm_coords.npy"
        pixel_coords_path = output_dir / "inflam_cell_pixel_coords.npy"

        output_dir.mkdir(exist_ok=True)

        if mm_coords_path.exists() and not force:
            self.logger.info(f"Coordinate file already exists at {mm_coords_path}, skipping detection.")
            mm_coords = np.load(mm_coords_path)
        else:
            mm_coords = run_inflammatory_cell_detection(
                wsi_path=wsi_path,
                output_dir=output_dir,
                model_path="detection/models/",
                visualise=visualise
            )
            np.save(mm_coords_path, mm_coords)

        self.logger.info(f"Detected {len(mm_coords)} inflam_cells")

        return {
            "wsi_name": wsi_name,
            "inflam_cell_coords_path": str(mm_coords_path),
            "inflam_cell_pixel_coords_path": str(pixel_coords_path) if pixel_coords_path.exists() else None
        }

    def run_stage3(self, wsi_path: str, force: bool = False) -> dict:
        self.logger.info("Running Stage 3: Quantification of inflammatory cells per tubule")

        paths = self.get_output_paths(wsi_path)

        if paths["counts_csv"].exists() and paths["quant_json"].exists() and not force:
            self.logger.info(f"Quantification already exists in {self.quantification_dir}. Skipping.")
            with open(paths["quant_json"]) as f:
                return json.load(f)

        if not paths["tubule_mask"].exists():
            raise FileNotFoundError("Required tubule mask not found.")

        tubule_mask = np.load(paths["tubule_mask"])
        coords_path = self.detection_dir / paths["wsi_name"] / "inflam_cell_mm_coords.npy"
        mm_coords = np.load(coords_path)

        # Convert mm → pixel coordinates
        MICRONS_PER_PIXEL = 0.24199951445730394
        cell_coords = (mm_coords * 1000 / MICRONS_PER_PIXEL).astype(np.int32)

        from tiffslide import TiffSlide
        slide = TiffSlide(wsi_path)
        self.logger.info(f"[Stage 3] WSI name: {paths['wsi_name']}")
        self.logger.info(f"[Stage 3] WSI size: {slide.dimensions}")
        self.logger.info(f"[Stage 3] Tubule mask shape: {tubule_mask.shape}")

        foci_mask = identify_foci(tubule_mask, min_distance=100)
        counts_df = count_cells_in_tubules(cell_coords, tubule_mask, foci_mask)

        save_counts_csv(counts_df, paths["counts_csv"])
        summary_stats = analyze_tubule_cell_distribution(counts_df)
        per_tubule_counts = dict(zip(counts_df['tubule_id'], counts_df['cell_count']))

        output = {
            "wsi_name": paths["wsi_name"],
            "total_inflam_cells": int(len(cell_coords)),
            "total_tubules": int(len(np.unique(tubule_mask)) - 1),
            "tubule_counts_csv": str(paths["counts_csv"]),
            "per_tubule_inflam_cell_counts": {int(k): int(v) for k, v in per_tubule_counts.items()},
            "summary_stats": convert_numpy_types(summary_stats)
        }

        with open(paths["quant_json"], "w") as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Saved structured summary to {paths['quant_json']}")
        return output

    def run_stage4(self, wsi_path: str, force: bool = False) -> dict:
        import pandas as pd

        self.logger.info("Running Stage 4: Grading")

        paths = self.get_output_paths(wsi_path)
        if paths["grading_report"].exists() and not force:
            self.logger.info("Grading already exists. Skipping.")
            with open(paths["grading_report"]) as f:
                return json.load(f)

        if not paths["counts_csv"].exists():
            raise FileNotFoundError("Required quantification CSV not found. Run Stage 3.")

        # Load the per-tubule DataFrame
        counts_df = pd.read_csv(paths["counts_csv"])

        # Calculate score using loaded counts
        grading_result = calculate_tubulitis_score(
            counts_df=counts_df,
            output_dir=self.grading_dir
        )

        return {
            "wsi_name": paths["wsi_name"],
            "tubulitis_score": grading_result["score"],
            "grading_report": grading_result["report_path"]
        }
        
    def run_by_stage(self, wsi_path: str, stage: str, force: bool = False, visualise: bool = False) -> Dict[str, Any]:
        if stage == "1":
            return self.run_stage1(wsi_path, force=force, visualise=visualise)
        elif stage == "2":
            return self.run_stage2(wsi_path, force=force, visualise=visualise)
        elif stage == "3":
            return self.run_stage3(wsi_path, force=force)
        elif stage == "4":
            return self.run_stage4(wsi_path, force=force)
        elif stage == "all":
            return self.run_pipeline(wsi_path, force=force, visualise=visualise)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def run_pipeline(self, wsi_path: str, force: bool = False, visualise: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Starting full pipeline for {wsi_path}")
        self.run_stage1(wsi_path, force=force, visualise=visualise)
        self.logger.info("Stage 1 completed. Proceeding to Stage 2.")
        self.run_stage2(wsi_path, force=force, visualise=visualise)
        self.logger.info("Stage 2 completed. Proceeding to Stage 3.")
        self.run_stage3(wsi_path, force=force)
        self.logger.info("Stage 3 completed. Proceeding to Stage 4.")
        return self.run_stage4(wsi_path, force=force)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="KidneyGrader: the E2E pipeline for Banff scoring of kidney biopsies")
    
    parser.add_argument("--input_path", required=True, help="Full path to the WSI image or patch")
    parser.add_argument("--output_dir", required=True, help="Directory where results will be saved")

    parser.add_argument("--stage", 
        choices=["1", "2", "3", "4", "all", "segment", "detect", "quantify", "grade"],
        default="all",
        help="Which stage to run (default: all). Can use stage number (1,2,3) or name (segment,quantify,grade)"
    )

    parser.add_argument("--model_path", type=str, 
        default="checkpoints/improved_unet_best.pth",
        help="Path to segmentation model checkpoint"
    )

    parser.add_argument("--force", action="store_true", help="Recompute all stages even if outputs exist")
    parser.add_argument("--visualise", action="store_true", help="Visualise segmentation results")

    args = parser.parse_args()

    console.print("[bold cyan]KidneyGrader Pipeline Starting...[/bold cyan]")
    console.print(f"[green]Input:[/green] {args.input_path}")
    console.print(f"[green]Stage:[/green] {args.stage}")
    console.print(f"[green]Output directory:[/green] {args.output_dir}")

    stage_map = {"segment": "1", "detect": "2", "quantify": "3", "grade": "4"}
    stage = stage_map.get(args.stage, args.stage)

    start = time.time()
    pipeline = KidneyGraderPipeline(output_dir=args.output_dir, model_path=args.model_path)
    result = pipeline.run_by_stage(args.input_path, stage, force=args.force, visualise=args.visualise)
    end = time.time()

    console.print(f"[bold green]Done in {end - start:.2f} seconds.[/bold green]")
    console.print(f"[bold yellow]Result:[/bold yellow] {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()
