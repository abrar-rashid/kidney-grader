import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops

def identify_foci(tubule_instance_mask, min_distance=100): 
    print("Identifying foci...")
    # Based on Banff tubulitis score specification, identifies spacial clusters of tubules, known as foci, which are within
    # min_distance of each other (centroid distance). Returns a new mask with unique foci labels/ids rather than tubules.

    centroids = []
    labels = []
    label_coords = {}


    props = regionprops(tubule_instance_mask) # fast centroid extraction

    for prop in tqdm(props, desc="Finding centroids"):
        label = prop.label
        centroid = prop.centroid  # (y, x)
        coords = prop.coords       # all pixels of the region
        centroids.append(centroid)
        labels.append(label)
        label_coords[label] = coords
        
    centroids = np.array(centroids)
    assigned = np.zeros(len(centroids), dtype=bool)
    foci_mask = np.zeros_like(tubule_instance_mask, dtype=np.uint16)
    current_focus = 1

    for i in tqdm(range(len(centroids)), desc="Clustering tubules into foci"):
        if assigned[i]:
            continue

        # perform breadth-first search to find all connected components
        queue = [i]
        assigned[i] = True
        current_focus_labels = [labels[i]]

        while queue:
            idx = queue.pop()
            dists = np.linalg.norm(centroids - centroids[idx], axis=1)
            neighbors = np.where((dists < min_distance) & (~assigned))[0]
            for n in neighbors:
                assigned[n] = True
                queue.append(n)
                current_focus_labels.append(labels[n])

        for label in current_focus_labels:
            coords = label_coords[label]
            foci_mask[coords[:, 0], coords[:, 1]] = current_focus
            
        current_focus += 1

    return foci_mask