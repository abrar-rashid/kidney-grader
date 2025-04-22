import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from grading.banff_grade import calculate_tubulitis_score
from quantification.quantify import analyze_tubule_cell_distribution, convert_numpy_types, count_cells_in_tubules, save_counts_csv
from quantification.tubule_utils import identify_foci

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
        
        # output dirs are stage-specific
        self.segmentation_dir = self.output_dir / "segmentation"
        self.detection_dir = self.output_dir / "detection"
        self.quantification_dir = self.output_dir / "quantification"
        self.grading_dir = self.output_dir / "grading"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segmentation_dir.mkdir(exist_ok=True)
        self.quantification_dir.mkdir(exist_ok=True)
        self.grading_dir.mkdir(exist_ok=True)

    
    def run_stage1(self, wsi_path: str) -> Dict[str, Any]:
        from segmentation.segment import run_segment
        # to run semantic segmentation of WSI
        logger = logging.getLogger(__name__)
        logger.info(f"Running Stage 1: Segmentation for {wsi_path}")

        wsi_name = Path(wsi_path).stem
        output_dir = self.segmentation_dir / wsi_name
        mask_npy_path = output_dir / f"{wsi_name}_full_mask.npy"
        mask_png_path = output_dir / f"{wsi_name}_full_mask.png"

        if mask_npy_path.exists():
            logger.info(f"Segmentation mask already exists at {mask_npy_path}, skipping segmentation.")
            instance_mask_paths = {
                int(p.stem.split("class")[-1]): str(p)
                for p in output_dir.glob(f"{wsi_name}_full_instance_mask_class*.npy")
            }
            return {
                "mask_path": str(mask_png_path),
                "mask_npy_path": str(mask_npy_path),
                "instance_mask_paths": instance_mask_paths
            }

        try:
            segmentation_result = run_segment(
                in_path=wsi_path,
                output_dir=self.segmentation_dir,
                model_path=self.model_path
            )

            mask_png_path = segmentation_result["mask_path"]
            mask_npy_path = mask_png_path.replace(".png", ".npy")

            if not os.path.exists(mask_npy_path):
                raise FileNotFoundError(f"Segmentation .npy mask not found at {mask_npy_path}")
            
            return {
                "mask_path": mask_png_path,
                "mask_npy_path": mask_npy_path,
                "instance_mask_paths": segmentation_result["instance_mask_paths"]
            }
        except Exception as e:
            logger.error(f"Segmentation stage failed: {str(e)}", exc_info=True)
            raise

    def run_stage2(self, wsi_path) -> dict:
        from detection.detect import run_inflammatory_cell_detection

        logger = logging.getLogger(__name__)
        logger.info("Running Stage 2: Inflammatory cell detection using InstanSeg + classifier")

        wsi_name = Path(wsi_path).stem
        output_dir = self.output_dir / "detection" / wsi_name
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = output_dir / "inflam_cell_instance_mask.npy"
        if mask_path.exists():
            logger.info(f"Inflammatory cell mask already exists at {mask_path}, skipping detection.")
            inflam_cell_mask = np.load(mask_path)
        else:
            inflam_cell_mask = run_inflammatory_cell_detection(
                wsi_path=wsi_path,
                output_dir=output_dir,
                model_path="detection/models"
            )
            np.save(mask_path, inflam_cell_mask)

        logger.info(f"Detected {np.unique(inflam_cell_mask).size - 1} inflam_cells")

        return {
            "wsi_name": wsi_name,
            "wsi_path": wsi_path,
            "inflam_cell_mask": inflam_cell_mask,
            "inflam_cell_mask_path": str(mask_path)
        }


    def run_stage3(self, wsi_path: str, tubule_mask_path: Path = None, inflam_cell_mask_path: Path = None) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("Running Stage 3: Quantification of inflammatory cells per tubule")

        wsi_name = Path(wsi_path).stem

        if tubule_mask_path is None:
            # default path for tubule mask (stage 1 output)
            tubule_mask_path = self.segmentation_dir / wsi_name / f"{wsi_name}_full_instance_mask_class1.npy"
        tubule_mask = np.load(tubule_mask_path)

        if inflam_cell_mask_path is None:
            # default path for inflammatory cell mask (stage 2 output)
            inflam_cell_mask_path = self.detection_dir / wsi_name / "inflam_cell_instance_mask.npy"
        inflam_cell_mask = np.load(inflam_cell_mask_path)

        # generate foci mask
        foci_mask = identify_foci(tubule_mask, min_distance=100)

        # inflammatory cell coordinates
        cell_coords = np.argwhere(inflam_cell_mask > 0)

        # full dataframe of counts
        counts_df = count_cells_in_tubules(cell_coords, tubule_mask, foci_mask)

        # save counts CSV but also json
        output_csv_path = self.quantification_dir / f"{wsi_name}_tubule_counts.csv"
        save_counts_csv(counts_df, output_csv_path)
        summary_stats = analyze_tubule_cell_distribution(counts_df)

        per_tubule_counts = dict(zip(counts_df['tubule_id'], counts_df['cell_count'])) #simpler count dict

        output = {
            "wsi_name": wsi_name,
            "total_inflam_cells": int(len(np.unique(inflam_cell_mask)) - 1),
            "total_tubules": int(len(np.unique(tubule_mask)) - 1),
            "tubule_counts_csv": str(output_csv_path),
            "per_tubule_inflam_cell_counts": {int(k): int(v) for k, v in per_tubule_counts.items()},
            "summary_stats": convert_numpy_types(summary_stats)
        }

        output_json_path = self.quantification_dir / f"{wsi_name}_quantification.json"
        with open(output_json_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved structured summary to {output_json_path}")
        return output

    def run_stage4(self, wsi_path) -> dict:
        from grading.banff_grade import calculate_tubulitis_score
        import pandas as pd

        logger = logging.getLogger(__name__)
        logger.info("Running Stage 4: Grading")

        if not wsi_path:
            raise ValueError("wsi_path must be provided to locate Stage 3 output")

        wsi_name = Path(wsi_path).stem
        counts_csv_path = self.quantification_dir / f"{wsi_name}_tubule_counts.csv"

        if not counts_csv_path.exists():
            raise FileNotFoundError(f"Expected counts CSV not found: {counts_csv_path}")

        # Load the per-tubule DataFrame
        counts_df = pd.read_csv(counts_csv_path)

        # Calculate score using loaded counts
        grading_result = calculate_tubulitis_score(
            counts_df=counts_df,
            output_dir=self.grading_dir
        )

        return {
            "wsi_name": wsi_name,
            "tubulitis_score": grading_result["score"],
            "grading_report": grading_result["report_path"]
        }
        
    def run_pipeline(self, wsi_path: str) -> Dict[str, Any]:
        # run the entire pipeline, comprising segmentation, quantification and grading
        # pass in an WSI, and get back a dictionary containing results from all stages
        logger = logging.getLogger(__name__)
        logger.info(f"Starting pipeline for {wsi_path}")
        
        try:
            self.run_stage1(wsi_path)
            self.run_stage2(wsi_path)
            self.run_stage3(wsi_path)
            final_result = self.run_stage4(wsi_path)
            
            logger.info(f"Pipeline completed successfully for {wsi_path}")
            return final_result
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

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

    args = parser.parse_args()

    stage_map = {
        "segment": "1",
        "detect": "2",
        "quantify": "3",
        "grade": "4"
    }
    stage = stage_map.get(args.stage, args.stage)

    try:
        pipeline = KidneyGraderPipeline(
            output_dir=args.output_dir,
            model_path=args.model_path
        )
        wsi_path = args.input_path

        if stage == "1": # run segmentation only
            pipeline.run_stage1(wsi_path)

        elif stage == "2":
            # can run stage 2 independent of stage 1 to test the detection model
            stage2_result = pipeline.run_stage2(wsi_path)
        # TODO fix logic for this or change to error for stage == 3 and 4
        elif stage == "3":
            # stage1_result = pipeline.run_stage1(wsi_path)
            # stage2_result = pipeline.run_stage2(wsi_path)
            pipeline.run_stage3(wsi_path)

        elif stage == "4":
            # stage1_result = pipeline.run_stage1(wsi_path)
            # stage2_result = pipeline.run_stage2(wsi_path)
            # stage3_result = pipeline.run_stage3(stage1_result, stage2_result)
            pipeline.run_stage4(wsi_path)

        else:  # default: all
            pipeline.run_pipeline(wsi_path)

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
