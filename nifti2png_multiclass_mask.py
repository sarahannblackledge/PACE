import SimpleITK as sitk
import cv2
import os

'''Inputs:
    fpath_nifti_4D: filepath to 4D nifti mask (no overlap)
    save_dir: filepath do directory where 2D pngs should be saved
    indices: list containing indices of individual organs to include in 2D png mask.
    index of last dimension corresponds to organ:
        0 = prostate
        1 = bladder
        2 = rectum
    example useage:
    fpath_nifti_4D = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/
    masks4D_no_overlap/NIHR_1_MR37.nii'
    indices = [0, 1, 2]'''


def nifti2png_multiclass_mask(fpath_nifti_4D, save_dir, *args):
    mask_sitk = sitk.ReadImage(fpath_nifti_4D)
    mask_4D = sitk.GetArrayFromImage(mask_sitk)

    #Extract base filename from fpath_nifti
    fname_orig = os.path.split(fpath_nifti_4D)[1]
    fname_base = fname_orig.rsplit('.', 1)[0]

    #Initialize combined mask as first index called of masks in 4D array
    combined_mask = mask_4D[:, :, :, args[0][0]]
    remaining_indices = args[0][1::]

    #iteratively sum desired masks (multiplied by a unique integer) to combined mask
    counter = 1
    for i in remaining_indices:
        counter = counter + 1
        mask_i = (mask_4D[:, :, :, i])*counter
        combined_mask = combined_mask + mask_i

    #Save each axial slice of combined mask as png
    for slice in range(0, combined_mask.shape[0]-1):
        ax_slice = combined_mask[slice, :, :].astype('uint8')
        fname_new = fname_base + '_' + str(slice) + '.png'
        cv2.imwrite(os.path.join(save_dir, fname_new), ax_slice)

#Single nifti example
'''fpath_nifti_4D = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks4D_no_overlap/NIHR_1_MR37.nii'
save_dir = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/pngs/all_masks'
indices = [0, 1, 2]
combined_mask = nifti2png_multiclass_mask(fpath_nifti_4D, save_dir, indices)'''

#Loop through directory
dir_4D_nifti = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks4D_no_overlap'
save_dir = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/pngs/all_masks_prostate_bladder'
indices = [0, 1]

for file in os.listdir(dir_4D_nifti):
    if file.endswith(".nii"):
        fpath_nifti_4D = os.path.join(dir_4D_nifti, file)
        nifti2png_multiclass_mask(fpath_nifti_4D, save_dir, indices)

