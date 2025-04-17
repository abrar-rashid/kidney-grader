import numpy as np

from segmentation.utils import get_binary_class_mask

def identify_foci(instance_mask, min_distance=100): 
    # Based on Banff tubulitis score specification, identifies spacial clusters of tubules, known as foci, which are within
    # min_distance of each other. Returns a new mask with the foci rather than tubules.

    centroids = []
    labels = []
    
    for label in range(1, np.max(instance_mask) + 1): # get centroids of tubules
        if label in instance_mask:
            y_coords, x_coords = np.where(instance_mask == label)
            if len(y_coords) > 0:
                centroid = np.mean([y_coords, x_coords], axis=1)
                centroids.append(centroid)
                labels.append(label)
    
    if not centroids:
        return np.zeros_like(instance_mask)
    
    centroids = np.array(centroids)

    foci_mask = np.zeros_like(instance_mask)
    current_focus = 1
    
    # group tubules into foci based on distance
    for i in range(len(centroids)):
        if foci_mask[instance_mask == labels[i]].max() > 0:
            continue
            
        current_tubules = [i] # start new focus with one tubule
        foci_mask[instance_mask == labels[i]] = current_focus
        
        # then add nearby tubules to current_tubules and assign these tubules to current focus
        for j in range(i + 1, len(centroids)):
            if foci_mask[instance_mask == labels[j]].max() > 0:
                continue
                
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_distance:
                current_tubules.append(j)
                foci_mask[instance_mask == labels[j]] = current_focus
        
        current_focus += 1
    
    return foci_mask