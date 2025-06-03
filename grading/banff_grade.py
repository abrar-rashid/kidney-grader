import json
import pandas as pd
import logging
from pathlib import Path

def calculate_tubulitis_score(counts_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # filter to include only tubules with mononuclear cells
    inflamed = counts_df[counts_df.cell_count > 0]

    # if no inflamed tubules are present, is t0
    if inflamed.empty or len(inflamed) < 2:
        score = "t0"
    else:
        # determine score based on the most inflamed tubule
        max_cell_count = inflamed["cell_count"].max()

        if max_cell_count > 10:
            score = "t3"
        elif max_cell_count >= 5:
            score = "t2"
        else:
            score = "t1"

    grading_report = {
        "Total inflammatory tubules": len(inflamed),
        "Max cells in any tubule": int(inflamed["cell_count"].max()) if not inflamed.empty else 0,
        "Tubulitis Grade": score
    }

    report_path = output_dir / "grading_report.json"
    with open(report_path, 'w') as f:
        json.dump(grading_report, f, indent=4)

    logging.info(f"Tubulitis grade computed: {score}")

    return {
        "score": score,
        "report_path": str(report_path)
    }
