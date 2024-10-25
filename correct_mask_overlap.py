import numpy as np
import SimpleITK as sitk
import os


def contour_check(mask):
    vals = np.unique(mask)
    if 1 in vals:
        output= True
    else:
        output= False
    return output


def convert_nifti_to_summed_arr(fpath_prostate, fpath_bladder, fpath_rectum):
    #Read nifti files and generate sitk objects
    prostate_sitk = sitk.ReadImage(fpath_prostate)
    bladder_sitk = sitk.ReadImage(fpath_bladder)
    rectum_sitk = sitk.ReadImage(fpath_rectum)

    #Get np arrays of image data
    prostate_mask = sitk.GetArrayFromImage(prostate_sitk)
    bladder_mask = sitk.GetArrayFromImage(bladder_sitk)
    rectum_mask = sitk.GetArrayFromImage(rectum_sitk)

    #Check that all masks actually have contours
    TF_prostate = contour_check(prostate_mask)
    if not TF_prostate:
        print('Mask empty for prostate in patient: ' + os.path.split(fpath_prostate)[1])

    TF_bladder = contour_check(bladder_mask)
    if not TF_bladder:
        print('Mask empty for bladder in patient: ' + os.path.split(fpath_rectum)[1])

    TF_rectum = contour_check(rectum_mask)
    if not TF_rectum:
        print('Mask empty for rectum in patient: ' + os.path.split(fpath_rectum)[1])

    if TF_prostate and TF_bladder and TF_rectum:
        print('All masks populated for patient: '  + os.path.split(fpath_prostate)[1])

    #Assign masks integer values: Prostate = 1, Bladder = 2, Rectum = 4
    bladder_mask = bladder_mask*2
    rectum_mask = rectum_mask*4

    #Sum 3D masks to create 3D multiclass mask
    multiclass_mask = prostate_mask + bladder_mask + rectum_mask
    print(np.unique(multiclass_mask))
    return multiclass_mask, prostate_sitk, bladder_sitk, rectum_sitk

def overlap_correct(multiclass_mask):
    #correct prostate-bladder overlap (if exists)
    overlap_indices3 = np.where(multiclass_mask == 3)
    multiclass_mask[overlap_indices3] = 1

    #correct for prostate-rectum overlap (if exists)
    overlap_indices5 = np.where(multiclass_mask == 5)
    multiclass_mask[overlap_indices5] = 1

    #correct for bladder-rectum overlap (if exists)
    overlap_indices6 = np.where(multiclass_mask == 6)
    multiclass_mask[overlap_indices6] = 2

    #Update label integers so consecutive (prostate = 1, bladder = 2, rectum = 3)
    label4 = np.where(multiclass_mask == 4)
    multiclass_mask[label4] = 3

    return multiclass_mask

def multiclass_to_nifti(multiclass_mask, template_sitk, save_subdirectory, fname):

    multiclass_sitk = sitk.GetImageFromArray(multiclass_mask.astype("uint8"))
    multiclass_sitk.CopyInformation(template_sitk)
    multiclass_sitk.SetMetaData("ContourName",  'multiclass')

    #Make subdirectory to save multi-class masks if doesn't already exist
    os.makedirs(save_subdirectory, exist_ok=True)

    #SAVE
    fpath_multiclass = os.path.join(save_subdirectory, fname)
    sitk.WriteImage(multiclass_sitk, fpath_multiclass, True)

    return
##########NEED TO FIX
def save_indiv(multiclass_mask, template_mask):
    #Separate corrected multiclass mask back into individual masks
    prostate_mask_corrected = np.zeros(prostate_mask.shape)
    bladder_mask_corrected = np.zeros(bladder_mask.shape)
    rectum_mask_corrected = np.zeros(rectum_mask.shape)

    prostate_indices = np.where(multiclass_mask == 1)
    prostate_mask_corrected[prostate_indices] = 1

    bladder_indices = np.where(multiclass_mask == 2)
    bladder_mask_corrected[bladder_indices] = 1

    rectum_indices = np.where(multiclass_mask == 4)
    rectum_mask_corrected[rectum_indices] = 1

    #Save as individual 3D masks (nifti)
    fname = os.path.split(fpath_prostate)[1]
    rootpath = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D_no_overlap'
    fpath_prostate_corrected = os.path.join(rootpath, 'ProstateOnly', fname)
    fpath_bladder_corrected = os.path.join(rootpath, 'Bladder', fname)
    fpath_rectum_corrected = os.path.join(rootpath, 'Rectum', fname)

    prostate_sitk_corrected = sitk.GetImageFromArray(prostate_mask_corrected.astype("uint8"))
    prostate_sitk_corrected.CopyInformation(prostate_sitk)

    bladder_sitk_corrected = sitk.GetImageFromArray(bladder_mask_corrected.astype("uint8"))
    bladder_sitk_corrected.CopyInformation(bladder_sitk)

    rectum_sitk_corrected = sitk.GetImageFromArray(rectum_mask_corrected.astype("uint8"))
    rectum_sitk_corrected.CopyInformation(rectum_sitk)

    sitk.WriteImage(prostate_sitk_corrected, fpath_prostate_corrected, True)
    sitk.WriteImage(bladder_sitk_corrected, fpath_bladder_corrected, True)
    sitk.WriteImage(rectum_sitk_corrected, fpath_rectum_corrected, True)

    #Concatenate into 4D mask and save (nifti)
    multiclass_mask_4D = np.concatenate((prostate_mask_corrected[...,np.newaxis],
                                         bladder_mask_corrected[..., np.newaxis],
                                         rectum_mask_corrected[..., np.newaxis]), axis=3)

    multiclass_sitk_corrected = sitk.GetImageFromArray(multiclass_mask_4D.astype("uint8"))
    multiclass_sitk_corrected.CopyInformation(prostate_sitk)
    multiclass_sitk_corrected.SetMetaData("ContourName", "Prostate_Bladder_Rectum")

    sitk.WriteImage(multiclass_sitk_corrected, os.path.join('/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks4D_no_overlap', fname), True)

    return


#############################################################
#Define folders
save_directory = '/Users/sblackledge/DATA/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump4'
save_subdirectory = os.path.join(save_directory, 'masks3D_PBR')
input_dir = 'masks3D'
input_dir_prostate = os.path.join(save_directory, 'masks3D', 'ProstateOnly')
input_dir_bladder = os.path.join(save_directory, 'masks3D', 'Bladder')
input_dir_rectum = os.path.join(save_directory, 'masks3D', 'Rectum')

#Loop through directory
for file in os.listdir(input_dir_prostate):
    if file.endswith(".nii.gz"):
        print(file)
        fpath_prostate = os.path.join(input_dir_prostate, file)
        fpath_bladder = os.path.join(input_dir_bladder, file)
        fpath_rectum = os.path.join(input_dir_rectum, file)

        multiclass_mask_uncorrected, prostate_sitk, bladder_sitk, rectum_sitk = convert_nifti_to_summed_arr(fpath_prostate, fpath_bladder, fpath_rectum)
        multiclass_mask_corrected = overlap_correct(multiclass_mask_uncorrected)
        multiclass_to_nifti(multiclass_mask_corrected, prostate_sitk, save_subdirectory, file)


