import SimpleITK as sitk
import numpy as np
import skimage


''' Uniformly expands structure by user-specified amount (mm)
inputs:
    1. fpath_nifti_mask: str - full file name of nifi file of mask of structure that should be uniformly expanded
    2. mm_expansion: int - number of millimeters to uniformly expand structure
outut:
    1. expanded_array: np array [slices, x, y] of expanded mask. '''
def uniform_expansion(fpath_nifti_mask, mm_expansion):
    im_sitk = sitk.ReadImage(fpath_nifti_mask)
    im_array = sitk.GetArrayFromImage(im_sitk)

    #Get resolution of sitk image
    px_spacing = np.asarray(im_sitk.GetSpacing()) #units of mm

    #Convert between mm_expansion to pixel expansion in each direction
    mm_expansion3d = np.array([mm_expansion, mm_expansion, mm_expansion])
    px_expansion = mm_expansion3d/px_spacing
    px_expansion_rounded = np.round(px_expansion)
    px_expansion_int = px_expansion_rounded.astype(int)

    #Uniformly expand array
    TF = np.all(px_expansion_int == px_expansion_int[0])

    if TF: #If expansion is the same number of pixels in each direction, can operate in 3D
        expanded_array = skimage.segmentation.expand_labels(im_array, distance=px_expansion[0])

    elif px_expansion_rounded[0] == px_expansion_rounded[1]: #if isotropic pixel size in axial orientation, but different slice thickness
        expanded_array = np.zeros(im_array.shape)
        #Expand in axial plan (lr and ap)
        SI_extent = []
        for slice in range(im_array.shape[0]):
            expanded_slice = skimage.segmentation.expand_labels(im_array[slice, :, :], distance=px_expansion[0])
            expanded_array[slice, :, :] = expanded_slice
            TF_2 = np.any(im_array[slice, :, :])
            SI_extent.append(TF_2)
        #Expand in Sup/inf plan
        orig_inds = [i for i, x in enumerate(SI_extent) if x]
        expanded_array[orig_inds[0]-1, :, :] = expanded_array[orig_inds[0], :, :]
        expanded_array[orig_inds[-1] + 1, :, :] = expanded_array[orig_inds[-1], :, :]

    return expanded_array





