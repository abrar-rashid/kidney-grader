import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

def setup_logging(output_dir: Path) -> None:
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
    
    def __init__(self, output_dir: str, model_path: str = "checkpoints/improved_unet_best.pth"):
        self.output_dir = Path(output_dir)
        self.model_path = model_path

        setup_logging(self.output_dir)
        
        # output dirs are stage-specific
        self.segmentation_dir = self.output_dir / "segmentation"
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
            
            segmentation_array = np.load(mask_npy_path)

            return {
                "wsi_name": wsi_name,
                "wsi_path": wsi_path,
                "segmentation_mask": segmentation_array,
                "mask_path": mask_png_path,
                "mask_npy_path": mask_npy_path,
            }
        except Exception as e:
            logger.error(f"Segmentation stage failed: {str(e)}", exc_info=True)
            raise
    
    def run_stage2(self, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        from quantification.detect_mononuclear_cells import detect_mononuclear_cells
        from quantification.tubule_utils import get_tubule_instances, identify_foci
        # to run inflammatory/mononuclear cell detection and quantification wrt a structure (tubules by default)
        logger = logging.getLogger(__name__)
        logger.info(f"Running Stage 2: Quantification for {stage1_result['wsi_name']}")
        
        try:
            cell_detection_result = detect_mononuclear_cells(
                wsi_path=stage1_result["wsi_path"]
            )
            
            tubule_mask, num_tubules = get_tubule_instances(
                mask=stage1_result["segmentation_mask"],
                tissue_type=1  # 1 = tubuli
            )
            
            foci_mask = identify_foci(tubule_mask) # clusters nearby tubules into foci for the tubulitis score calcualtion
            
            return {
                **stage1_result,
                "cell_coordinates": cell_detection_result["coordinates"],
                "cell_visualization": cell_detection_result["visualization_path"],
                "tubule_mask": tubule_mask,
                "num_tubules": num_tubules,
                "foci_mask": foci_mask
            }
        except Exception as e:
            logger.error(f"Quantification stage failed: {str(e)}", exc_info=True)
            raise
    
    def run_stage3(self, stage2_result: Dict[str, Any]) -> Dict[str, Any]:
        from grading.banff_grade import calculate_tubulitis_score
        # compute the final Banff grading for the tubulitis score
        logger = logging.getLogger(__name__)
        logger.info(f"Running Stage 3: Grading for {stage2_result['wsi_name']}")
        
        try:
            grading_result = calculate_tubulitis_score(
                cell_coordinates=stage2_result["cell_coordinates"],
                tubule_mask=stage2_result["tubule_mask"],
                foci_mask=stage2_result["foci_mask"],
                output_dir=self.grading_dir
            )
            
            return {
                **stage2_result,
                "tubulitis_score": grading_result["score"],
                "grading_report": grading_result["report_path"]
            }
        except Exception as e:
            logger.error(f"Grading stage failed: {str(e)}", exc_info=True)
            raise
    
    def run_pipeline(self, wsi_path: str) -> Dict[str, Any]:
        # run the entire pipeline, comprising segmentation, quantification and grading
        # pass in an WSI, and get back a dictionary containing results from all stages
        logger = logging.getLogger(__name__)
        logger.info(f"Starting pipeline for {wsi_path}")
        
        try:
            stage1_result = self.run_stage1(wsi_path)
            stage2_result = self.run_stage2(stage1_result)
            final_result = self.run_stage3(stage2_result)
            
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
        choices=["1", "2", "3", "all", "segment", "quantify", "grade"],
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
        "quantify": "2",
        "grade": "3"
    }
    stage = stage_map.get(args.stage, args.stage)

    try:
        pipeline = KidneyGraderPipeline(
            output_dir=args.output_dir,
            model_path=args.model_path
        )

        wsi_path = args.input_path

        if stage == "1":
            pipeline.run_stage1(wsi_path)
        elif stage == "2":
            stage1_result = pipeline.run_stage1(wsi_path)
            pipeline.run_stage2(stage1_result)
        elif stage == "3":
            stage1_result = pipeline.run_stage1(wsi_path)
            stage2_result = pipeline.run_stage2(stage1_result)
            pipeline.run_stage3(stage2_result)
        else:  # default: all
            pipeline.run_pipeline(wsi_path)

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
