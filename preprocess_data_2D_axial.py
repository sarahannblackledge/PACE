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

'''inputs:
    1. rescaled_arr: 3D numpy array in desired grayscale vals
    2. save_dir: str - directory where 2D image files should be saved.
    3. fpath_nifti: str - full filename of original nifti file to be converted into 2D image files
    4. orientation: str - either 'axial' or 'sagittal'. Indicates which orientation images should be saved in.'''

def arr_to_png(rescaled_arr, save_dir, fpath_nifti, orientation):

    #get first part of filename
    f1 = Path(fpath_nifti).stem

    if orientation == 'axial':

        for i in range(rescaled_arr.shape[0]):
            im_slice = rescaled_arr[i, :, :].astype('uint8')
            savename = f1 + '_' + str(i) + '.tiff'
            savepath = os.path.join(save_dir, savename)

            cv2.imwrite(savepath, im_slice)
            #np.save(savepath, im_slice)

    elif orientation == 'sagittal':

        for i in range(rescaled_arr.shape[2]):
            im_slice = rescaled_arr[:, :, i].astype('uint8')
            savename = f1 + '_' + str(i) + '.tiff'
            savepath = os.path.join(save_dir, savename)

            cv2.imwrite(savepath, im_slice)

        #display
        '''if i == 150:
            plt.figure()
            plt.imshow(im_slice, cmap='gray')
            plt.show()'''


# Single file example
'''fpath_nifti = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/MRI/NIHR_1_MR11.nii'
save_dir_ims = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/pngs/all_images'

# images
rescaled_arr = normalize_im(fpath_nifti, 'z-score')
arr_to_png(rescaled_arr, save_dir_ims, fpath_nifti)'''

#masks
'''fpath_nifti_mask = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/masks_multiclass/NIHR_1_MR97.nii'
save_dir_masks = '/Users/sblackledge/Documents/ProKnow_database/axial_data/all_masks'
rescaled_mask = normalize_im(fpath_nifti_mask, 'None')
arr_to_png(rescaled_mask, save_dir_masks, fpath_nifti_mask)'''


#Loop through all images for one patient
save_dir_ims = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/tiffs/all_sag_masks_SVs'
mr_dir = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D/SVsOnly'
for file in os.listdir(mr_dir):
    if file.endswith(".nii"):
        fpath_nifti = os.path.join(mr_dir, file)
        rescaled_arr = normalize_im(fpath_nifti, 'None')
        arr_to_png(rescaled_arr, save_dir_ims, fpath_nifti, 'sagittal')



