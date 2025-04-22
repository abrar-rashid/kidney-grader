import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

def count_cells_in_tubules(nuclei_coords, instance_mask, foci_mask):
    # for each tubule, record its cell count, position, and ids of itself and the focus it is assigned to.

    if len(nuclei_coords) == 0:
        return pd.DataFrame(columns=['tubule_id', 'x', 'y', 'cell_count', 'focus_id'])

    nuclei_coords = nuclei_coords.astype(int)
    nuclei_y, nuclei_x = nuclei_coords[:, 0], nuclei_coords[:, 1]

    # get tubule label for each nucleus
    tubule_ids = instance_mask[nuclei_y, nuclei_x]
    valid_mask = tubule_ids > 0
    tubule_ids = tubule_ids[valid_mask]
    nuclei_y = nuclei_y[valid_mask]
    nuclei_x = nuclei_x[valid_mask]

    # count how many cells fall in each tubule
    unique_ids, counts = np.unique(tubule_ids, return_counts=True)
    id_to_count = dict(zip(unique_ids, counts))

    # build results
    results = []
    for tub_id in tqdm(unique_ids, desc="Counting cells in tubules"):
        tubule_mask = instance_mask == tub_id
        y_coords, x_coords = np.where(tubule_mask)
        if len(y_coords) == 0:
            continue
        centroid_y = y_coords.mean()
        centroid_x = x_coords.mean()
        focus_id = foci_mask[y_coords[0], x_coords[0]]  # can also be refined

        results.append({
            'tubule_id': int(tub_id),
            'x': centroid_x,
            'y': centroid_y,
            'cell_count': int(id_to_count[tub_id]),
            'focus_id': int(focus_id)
        })

    df = pd.DataFrame(results)
    return df.sort_values("cell_count", ascending=False)

def save_counts_csv(counts_df, output_path): # saves results dataframe into CSV file 
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    counts_df.to_csv(output_path, index=False)
    
    # add summary statistics using the analysis function
    summary = analyze_tubule_cell_distribution(counts_df)
    
    summary_path = output_path.parent / "summary_stats.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    
    logging.info(f"saved cell counts to {output_path}")
    logging.info(f"summary statistics saved to {summary_path}")

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

def analyze_tubule_cell_distribution(counts_df):
    # analyses overall distribution and computes aggregate statistics 
    if counts_df.empty:
        return {}

    stats = {
        "total_tubules": len(counts_df),
        "total_cells": counts_df['cell_count'].sum(),
        "mean_cells_per_tubule": counts_df['cell_count'].mean(),
        "std_cells_per_tubule": counts_df['cell_count'].std(),
        "max_cells_in_tubule": counts_df['cell_count'].max(),
        "total_foci": len(np.unique(counts_df['focus_id']))
    }
    
    # stats for foci for the grading stage
    focus_stats = counts_df.groupby('focus_id').agg({
        'cell_count': ['count', 'sum', 'mean', 'max']
    }).reset_index()
    
    focus_stats.columns = ['focus_id', 'num_tubules', 'total_cells', 
                          'mean_cells_per_tubule', 'max_cells_in_tubule']
    
    stats['focus_stats'] = focus_stats.to_dict('records')
    
    stats['cell_count_distribution'] = counts_df['cell_count'].value_counts().sort_index().to_dict()
    
    return stats

