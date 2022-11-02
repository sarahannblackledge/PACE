import SimpleITK as sitk
import numpy as np
import os

''' Uniformly expands structure by user-specified amount (mm)
inputs:
    1. fpath_nifti_mask: str - full file name of nifi file of mask of structure that should be uniformly expanded
    2. mm_expansion: int - number of millimeters to uniformly expand structure
    3. save_flag: 0 if you don't want to save, 1 if you want to write as nifti
outut:
    1. expanded_array: np array [slices, x, y] of expanded mask. '''



def uniform_expansion(fpath_nifti_mask, mm_expansion, save_flag):
    im_sitk = sitk.ReadImage(fpath_nifti_mask)
    im_array = sitk.GetArrayFromImage(im_sitk)

    #Get resolution of sitk image
    px_spacing = np.asarray(im_sitk.GetSpacing()) #units of mm

    #Convert between mm_expansion to pixel expansion in each direction
    mm_expansion3d = np.array([mm_expansion, mm_expansion, mm_expansion])
    px_expansion = mm_expansion3d/px_spacing
    px_expansion_rounded = np.round(px_expansion)
    px_expansion_int = (px_expansion_rounded.astype(int)).tolist()

    #Dilate using sitk
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetKernelRadius((px_expansion_int[0], px_expansion_int[1], px_expansion_int[2]))
    dilated_mask = dilate_filter.Execute(im_sitk)

    #Write as nifti [optional]
    if save_flag == 1:
        orig_path = os.path.splitext(fpath_nifti_mask)[0]
        new_path = orig_path + '_expanded.nii'
        sitk.WriteImage(dilated_mask, new_path)

    return dilated_mask


#fpath_nifti_mask = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D/ProstateOnly/NIHR_1_MR11.nii'
#mm_expansion = 10
#dilated_mask = uniform_expansion(fpath_nifti_mask, mm_expansion, 1)



