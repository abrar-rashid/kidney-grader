import json
import pandas as pd
import logging
from pathlib import Path

def calculate_tubulitis_score(counts_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # filter to include only tubules with mononuclear cells
    inflamed = counts_df[counts_df.cell_count > 0]
    num_foci = 0
    
    # if no inflamed tubules are present, is t0
    if inflamed.empty:
        score = "t0"
    else:
        # group by foci and find max cell count per focus
        foci_max = inflamed.groupby("focus_id")["cell_count"].max()
        num_foci = len(foci_max)

        # check if there iss only one focus and no other inflamed tubules
        if num_foci == 1:
            # ensure no other inflamed tubules outside this focus
            total_inflamed_tubules = len(inflamed)
            if total_inflamed_tubules == inflamed['focus_id'].value_counts().iloc[0]:
                score = "t0"
            else:
                score = "t1"
        elif num_foci < 2:
            score = "t0"
        elif (foci_max > 10).any():
            score = "t3"
        elif (foci_max >= 5).any():
            score = "t2"
        else:
            score = "t1"

    grading_report = {
        "Total inflammatory tubules": len(inflamed),
        "Total foci": num_foci,
        "Max cells in any tubule": int(foci_max.max()) if not inflamed.empty else 0,
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