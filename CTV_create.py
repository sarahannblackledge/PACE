import SimpleITK as sitk
import numpy as np
import sys
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from uniform_expansion import uniform_expansion


def CTVpsv_4000_create(fpath_nifti_prostate, fpath_nifti_sv, mm_expansion):

    #Create sitk image objects
    prostate_sitk = sitk.ReadImage(fpath_nifti_prostate)
    sv_sitk = sitk.ReadImage(fpath_nifti_sv)

    # Convert into numpy arrays
    prostate_arr = sitk.GetArrayFromImage(prostate_sitk)
    sv_arr = sitk.GetArrayFromImage(sv_sitk)

    #Expand prostate contour
    prostate_expanded = uniform_expansion(fpath_nifti_prostate, mm_expansion, 0)
    prostate_expanded_arr = sitk.GetArrayFromImage(prostate_expanded)

    #Calculate CTVpsv_4000
    P_CTVsv_union = prostate_arr + sv_arr
    P_CTVsv_union[P_CTVsv_union > 1] = 1 #convert areas of overlap to '1'
    CTVpsv_4000_arr = P_CTVsv_union + prostate_expanded_arr
    CTVpsv_4000_arr[CTVpsv_4000_arr < 2] = 0
    CTVpsv_4000_arr[CTVpsv_4000_arr > 0] = 1

    #Create sitk image object of CTVpsv4000
    CTVpsv_4000_sitk = sitk.GetImageFromArray(CTVpsv_4000_arr)
    CTVpsv_4000_sitk.CopyInformation(prostate_sitk)


    return CTVpsv_4000_sitk

'''mm_expansion = 10
fpath_nifti_prostate = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D_no_overlap/ProstateOnly/NIHR_1_MR11.nii'
fpath_nifti_sv = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D/SVsOnly/NIHR_1_MR11.nii'

CTVpsv_4000_sitk = CTVpsv_4000_create(fpath_nifti_prostate, fpath_nifti_sv, mm_expansion)

sitk.WriteImage(CTVpsv_4000_sitk, '/Users/sblackledge/Desktop/test.nii', True)'''






