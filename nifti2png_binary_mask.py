import SimpleITK as sitk
import os
import numpy as np
import cv2

'''Inputs:
    fpath_nifti_3D: filepath to 3D nifti mask
    save_dir: filepath do directory where 2D pngs should be saved


    example useage:
    fpath_nifti_4D = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/
    masks4D_no_overlap/NIHR_1_MR37.nii'
    indices = [0, 1, 2]'''


def nifti2png_multiclass_mask(fpath_nifti_3D, save_dir):
    mask_sitk = sitk.ReadImage(fpath_nifti_3D)
    mask_3D = sitk.GetArrayFromImage(mask_sitk)

    #Extract base filename from fpath_nifti
    fname_orig = os.path.split(fpath_nifti_3D)[1]
    fname_base = fname_orig.rsplit('.', 1)[0]

    #Save each axial slice of binary mask as png
    for slice in range(0, mask_3D.shape[0]):
        ax_slice = mask_3D[slice, :, :].astype('uint8')
        fname_new = fname_base + '_' + str(slice) + '.tiff'
        fpath_new = os.path.join(save_dir, fname_new)
        cv2.imwrite(fpath_new, ax_slice)


#Single nifti example
'''fpath_nifti_4D = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks4D_no_overlap/NIHR_1_MR37.nii'
save_dir = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/pngs/all_masks_prostate_bladder_rectum'
indices = [0, 1, 2]
combined_mask = nifti2png_multiclass_mask(fpath_nifti_4D, save_dir, indices)'''

#Loop through directory
dir_3D_nifti = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D/SVsOnly'
save_dir = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/tiffs/all_masks_SVs'

for file in os.listdir(dir_3D_nifti):
    if file.endswith(".nii"):
        fpath_nifti_3D = os.path.join(dir_3D_nifti, file)
        nifti2png_multiclass_mask(fpath_nifti_3D, save_dir)

