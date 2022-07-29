import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def normalize_im(fpath_nifti, norm_type):
    mr_sitk = sitk.ReadImage(fpath_nifti)
    arr = sitk.GetArrayFromImage(mr_sitk)

    if 'z-score' in norm_type:
        #Perform z-score normalization
        mu = np.mean(arr)
        sigma = np.std(arr)
        normalized_arr = (arr - mu)/sigma

        # Rescale to 0-255 for saving to png
        rescaled_arr = ((normalized_arr - normalized_arr.min()) *
                        (1 / (normalized_arr.max() - normalized_arr.min()) * 255)).astype('uint8')


    if 'None' in norm_type:
        #Don't rescale. pngs will look completely black, but segmentations are there.
        rescaled_arr = arr


    return(rescaled_arr)

def arr_to_png(rescaled_arr, save_dir, fpath_nifti):
    #transpose to conventional slice order
    arr = np.array(rescaled_arr).transpose((2, 1, 0))
    arr = np.rot90(arr, k=-1, axes=(0, 1))

    #get first part of filename
    f1 = Path(fpath_nifti).stem

    for i in range(arr.shape[-1]):
        ax_slice = arr[:, :, i].astype('uint8')
        savename = f1 + '_' + str(i) + '.png'
        savepath = os.path.join(save_dir, savename)

        cv2.imwrite(savepath, ax_slice)

        #display
        '''if i == 150:
            plt.figure()
            plt.imshow(ax_slice, cmap='gray')
            plt.show()'''

#Loop through all images for one patientß
save_dir_ims = '/Users/sblackledge/Documents/ProKnow_database/axial_data/all_images'
save_dir_masks = '/Users/sblackledge/Documents/ProKnow_database/axial_data/all_masks'

#adjust per image-dose pair
patient = 'NIHR_14'
ids = [21, 23, 34, 47, 57, 82, 94, 14]

for id in ids:

    #hard-coded variables (consistent in Proknow database directory)
    id_str = 'MR' + str(id)
    fpath_nifti = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/MRI/' + patient + '_' + id_str + '.nii'
    fpath_nifti_mask = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/masks_multiclass/' + patient + '_' + id_str + '.nii'
    id_num = str(id)
    if not os.path.isfile(fpath_nifti):
        id_str = 'RefMR' + str(id)
        fpath_nifti = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/MRI/' + patient + '_' + id_str + '.nii'
        fpath_nifti_mask = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/masks_multiclass/' + patient + '_' + id_str + '.nii'
        id_num = id_str

    #images
    rescaled_arr = normalize_im(fpath_nifti, 'z-score')
    arr_to_png(rescaled_arr, save_dir_ims, fpath_nifti)

    #masks
    rescaled_mask = normalize_im(fpath_nifti_mask, 'None')
    arr_to_png(rescaled_mask, save_dir_masks, fpath_nifti_mask)




# Single file example
'''fpath_nifti = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/MRI/NIHR_1_MR97.nii'
fpath_nifti_mask = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/masks_multiclass/NIHR_1_MR97.nii'
save_dir_ims = '/Users/sblackledge/Documents/ProKnow_database/axial_data/all_images'
save_dir_masks = '/Users/sblackledge/Documents/ProKnow_database/axial_data/all_masks'

# images
rescaled_arr = normalize_im(fpath_nifti, 'z-score')
arr_to_png(rescaled_arr, save_dir_ims, fpath_nifti)

#masks
rescaled_mask = normalize_im(fpath_nifti_mask, 'None')
arr_to_png(rescaled_mask, save_dir_masks, fpath_nifti_mask)'''
