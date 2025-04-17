import numpy as np

def count_inflam_cells_per_tubule(tubule_mask: np.ndarray, inflam_cell_mask: np.ndarray) -> dict:
    lymph_ids = np.unique(inflam_cell_mask)
    lymph_ids = lymph_ids[lymph_ids != 0]

    tubule_ids = np.unique(tubule_mask)
    tubule_ids = tubule_ids[tubule_ids != 0]

    per_tubule_counts = {}
    for tubule_id in tubule_ids:
        tubule_bin = (tubule_mask == tubule_id)
        count = sum(
            np.logical_and(tubule_bin, (inflam_cell_mask == lymph_id)).sum() > 0
            for lymph_id in lymph_ids
        )
        per_tubule_counts[tubule_id] = count

    return per_tubule_counts
