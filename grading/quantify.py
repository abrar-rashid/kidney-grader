import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from skimage.measure import regionprops_table

from skimage.measure import regionprops

def count_cells_in_tubules(cell_coords, instance_mask):
    # for each tubule, record its cell count, position, and id

    if len(cell_coords) == 0:
        return pd.DataFrame(columns=['tubule_id', 'x', 'y', 'cell_count'])

    cell_coords = cell_coords.astype(int)
    cells_y, cells_x = cell_coords[:, 0], cell_coords[:, 1]

    h, w = instance_mask.shape
    valid_mask = (cells_y >= 0) & (cells_y < h) & (cells_x >= 0) & (cells_x < w)
    cells_y = cells_y[valid_mask]
    cells_x = cells_x[valid_mask]

    # get tubule label for each nucleus
    tubule_ids = instance_mask[cells_y, cells_x]
    valid_mask = tubule_ids > 0
    tubule_ids = tubule_ids[valid_mask]
    cells_y = cells_y[valid_mask]
    cells_x = cells_x[valid_mask]

    # count how many cells fall in each tubule
    unique_ids, counts = np.unique(tubule_ids, return_counts=True)
    id_to_count = dict(zip(unique_ids, counts))

    # optimized: Get properties for all regions at once using regionprops_table
    props = regionprops_table(instance_mask, properties=['label', 'centroid'])
    tubule_labels = props['label']
    centroids_y = props['centroid-0']
    centroids_x = props['centroid-1']

    # filter to only include tubules that have cells
    mask = np.isin(tubule_labels, list(id_to_count.keys()))
    filtered_labels = tubule_labels[mask]
    filtered_y = centroids_y[mask]
    filtered_x = centroids_x[mask]

    # build results using vectorized operations
    results = []
    for i, tub_id in enumerate(filtered_labels):
        results.append({
            'tubule_id': int(tub_id),
            'x': filtered_x[i],
            'y': filtered_y[i],
            'cell_count': int(id_to_count[tub_id])
        })

    if not results:
        return pd.DataFrame(columns=['tubule_id', 'x', 'y', 'cell_count'])

    return pd.DataFrame(results).sort_values("cell_count", ascending=False)


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
        return {
            "total_tubules": 0,
            "total_cells":   0,
            "mean_cells_per_tubule": 0.0,
            "std_cells_per_tubule":  0.0,
            "max_cells_in_tubule":   0,
            "cell_count_distribution": {},
            "mean_cells_top_1pct":  0.0,
            "mean_cells_top_5pct":  0.0,
            "mean_cells_top_10pct": 0.0,
        }

    stats = {
        "total_tubules": len(counts_df),
        "total_cells":   counts_df["cell_count"].sum(),
        "mean_cells_per_tubule": counts_df["cell_count"].mean(),
        "std_cells_per_tubule":  counts_df["cell_count"].std(),
        "max_cells_in_tubule":   counts_df["cell_count"].max(),
        "cell_count_distribution":
            counts_df["cell_count"].value_counts().sort_index().to_dict(),
    }

    stats["mean_cells_top_1pct"]  = _mean_cells_in_top_percent(counts_df, 0.01)
    stats["mean_cells_top_5pct"]  = _mean_cells_in_top_percent(counts_df, 0.05)
    stats["mean_cells_top_10pct"] = _mean_cells_in_top_percent(counts_df, 0.10)

    return convert_numpy_types(stats)

def _mean_cells_in_top_percent(df: pd.DataFrame, percent: float) -> float:
    # return the mean cell count among the top *percent* of tubules.
    if df.empty:
        return 0.0
    n_top = max(1, int(np.ceil(len(df) * percent)))
    return df.nlargest(n_top, "cell_count")["cell_count"].mean()
