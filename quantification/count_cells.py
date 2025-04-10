import numpy as np
import pandas as pd
from pathlib import Path
import logging

def count_cells_in_tubules(nuclei_coords, instance_mask, foci_mask):
    # for each tubule, record its cell count, position, and ids of itself and the focus it is assigned to.

    if len(nuclei_coords) == 0:
        return pd.DataFrame(columns=['tubule_id', 'x', 'y', 'cell_count', 'focus_id'])
    
    nuclei_coords = nuclei_coords.astype(int)
    
    results = []
    
    for label in range(1, np.max(instance_mask) + 1): 
        if label not in instance_mask:
            continue
        tubule_mask = (instance_mask == label)
        
        cells_in_tubule = 0
        for y, x in nuclei_coords:
            if tubule_mask[y, x]:
                cells_in_tubule += 1
        if cells_in_tubule > 0:
            y_coords, x_coords = np.where(tubule_mask)
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            
            focus_id = foci_mask[y_coords[0], x_coords[0]]
            
            results.append({
                'tubule_id': label,
                'x': centroid_x,
                'y': centroid_y,
                'cell_count': cells_in_tubule,
                'focus_id': focus_id
            })
    
    df = pd.DataFrame(results)    
    df = df.sort_values('cell_count', ascending=False)
    
    return df

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

