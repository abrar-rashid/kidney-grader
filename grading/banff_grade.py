import pandas as pd
import logging
from pathlib import Path

def calculate_tubulitis_score(counts_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if counts_df.empty:
        score = "t0"
    else:
        foci_max = counts_df.groupby("focus_id")["cell_count"].max()
        num_foci = len(foci_max)

        if num_foci < 2:
            score = "t0"
        elif (foci_max > 10).any():
            score = "t3"
        elif (foci_max >= 5).any():
            score = "t2"
        else:
            score = "t1"

    report_path = output_dir / "grading_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Banff Tubulitis Grading Report\n")
        f.write(f"------------------------\n")
        f.write(f"Total inflammatory tubules: {len(counts_df)}\n")
        f.write(f"Total foci: {counts_df['focus_id'].nunique()}\n")
        f.write(f"Max cells in any tubule: {counts_df['cell_count'].max() if not counts_df.empty else 0}\n")
        f.write(f"Tubulitis Grade: {score}\n")

    logging.info(f"Tubulitis grade computed: {score}")
    
    return {
        "score": score,
        "report_path": str(report_path)
    }