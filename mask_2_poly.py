import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as region_label

''' Converts mask into list of contours (one contour per mask region).
inputs:
    1. mask_array: 2D numpy array
outputs:
    1. contours: list of contours bounding each distinct region within mask_array. Contours are described by xy 
    coordinate points (float).

'''

def mask_2_poly(mask_array):
    mask_array = mask_array > 0
    if np.sum(mask_array) == 0:
        return None
    labels, n_labels = region_label(mask_array)
    contours = []
    for label in range(n_labels):
        f = plt.figure()
        cs = plt.contour(labels == label + 1, [0.5])
        verts = cs.collections[0]._paths[0].vertices
        plt.close()
        contours.append(verts)
    return contours