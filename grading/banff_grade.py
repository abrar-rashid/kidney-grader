import pandas as pd
import logging
from pathlib import Path
from quantification.count_cells import count_cells_in_tubules, save_counts_csv

def calculate_tubulitis_score(cell_coordinates, tubule_mask, foci_mask, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_df = count_cells_in_tubules(cell_coordinates, tubule_mask, foci_mask)

    output_csv_path = output_dir / "tubulitis_counts.csv"
    save_counts_csv(counts_df, output_csv_path)

    if counts_df.empty:
        score = "t0"
    else:
        max_cells = counts_df['cell_count'].max()
        if max_cells <= 4:
            score = "t1"
        elif 5 <= max_cells <= 10:
            score = "t2"
        elif max_cells > 10:
            score = "t3"
        else:
            score = "Undefined"

    report_path = output_dir / "grading_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Banff Tubulitis Grading Report\n")
        f.write(f"------------------------\n")
        f.write(f"max cells in any tubule: {max_cells if not counts_df.empty else 0}\n")
        f.write(f"total tubules with inflammation: {len(counts_df)}\n")
        f.write(f"total foci: {len(counts_df['focus_id'].unique()) if not counts_df.empty else 0}\n")
        f.write(f"Tubulitis Grade: {score}\n")

    logging.info(f"Tubulitis grade computed: {score}")
    
    return {
        "score": score,
        "report_path": str(report_path)
    }
