import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import re
import SimpleITK as sitk

'''Hard-code class values
0 = background, 1 = prostate, 2 = bladder, 3 = rectum'''
class_vals = np.array([0, 1, 2, 3])

def multiclass_to_indiv(im2D, class_vals, index):
    mask = copy.deepcopy(im2D)
    mask[mask == class_vals[index]] = 10
    mask[mask < 10] = 0
    mask[mask == 10] = 1
    return mask

'''inputs: 
    1. png_dir: str - full filepath to directory where prediction image files (png, tiff, etc.) are stored.
    2. fraction_name: str - e.g. 'NIHR_1_MR11'.
    3. index: int - indicates which class to make mask from. 1 for single-class segmentations
    4. orientation: str- either 'axial' or 'sagittal'. 
    '''
def array_from_png(png_dir, fraction_name, index, orientation):
    file_im = np.array([os.path.join(png_dir, fl) for fl in os.listdir(png_dir) if fraction_name in fl])
    file_list = list(file_im)
    locations = []

    #Sort
    for f in file_list:
        inds_ = [m.start() for m in re.finditer('_', f)][-1]
        location = int(f[(inds_+1):-5])
        locations.append(location)

    file_im = file_im[np.argsort(locations)]
    file_im = np.squeeze(file_im)

    #Append in 3D
    mask_3D = []
    for file in file_im:
        im2D = plt.imread(file)

        #Make individual 2D masks
        mask_2D = multiclass_to_indiv(im2D, class_vals, index)
        mask_3D.append(mask_2D)

    #Convert to np
    if orientation=='axial':
        mask_np = np.array(mask_3D)

    elif orientation=='sagittal':
        mask_np = np.array(mask_3D)
        mask_np = mask_np.transpose((1, 2, 0))

    return mask_np


#Save to nifti using ground-truth template file
def np_to_nifti(template_nifti_fpath, mask_np, save_path):
    mask_image_sub = sitk.GetImageFromArray(mask_np.astype("uint8"))
    template = sitk.ReadImage(template_nifti_fpath)
    template_np = sitk.GetArrayFromImage(template)
    mask_image_sub.CopyInformation(template)
    sitk.WriteImage(mask_image_sub, save_path)

#Extract unique fraction names (i.e. 'NIHR_1_MR11') from png directory
def get_fraction_names(png_dir):
    new_list = []
    for file in os.listdir(png_dir):
        inds_ = [m.start() for m in re.finditer('_', file)][-1]
        frac_name = file[0:inds_]
        if frac_name not in new_list:
            new_list.append(frac_name)

    return new_list


png_dir = '/Users/sblackledge/Documents/MRL_segmentation_predictions/SVs/attempt4/pred_tiffs_val'
index = 1
orientation = 'axial'
template_nifti_root= '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D/SVsOnly'

#EDIT!
save_root = '/Users/sblackledge/Documents/MRL_segmentation_predictions/SVs/attempt4/niftis_validation'


frac_list = get_fraction_names(png_dir)
for frac_name in frac_list:
    #Extract 3D numpy array for specified organ (given by index)
    mask_np = array_from_png(png_dir, frac_name, index, orientation)
    save_dir = os.path.join(save_root, frac_name)

    #Create directory for each fraction
    isdir = os.path.isdir(save_dir)
    if not isdir:
        os.mkdir(save_dir)

    if index == 1:
        #template_nifti_dir = os.path.join(template_nifti_root, 'ProstateOnly')
        template_nifti_dir = template_nifti_root
        save_path = os.path.join(save_dir, 'SVs_pred.nii')

    elif index == 2:
        template_nifti_dir = os.path.join(template_nifti_root, 'Bladder')
        save_path = os.path.join(save_dir, 'Bladder_pred.nii')

    elif index == 3:
        template_nifti_dir = os.path.join(template_nifti_root, 'Rectum')
        save_path = os.path.join(save_dir, 'Rectum_pred.nii')

    fname = frac_name + '.nii'
    template_nifti_path = os.path.join(template_nifti_dir, fname)

    #Save as nifti
    np_to_nifti(template_nifti_path, mask_np, save_path)




