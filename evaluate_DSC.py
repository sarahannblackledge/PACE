import numpy as np
import SimpleITK as sitk
import os
import sys
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from DSC_calc import DSC_calc

'''index keys:
    1 - prostate
    2 - bladder
    3 - rectum'''

def export_DSCs(index, dir_truth, dir_preds):
    DSC = []

    #Identify which predictions you'd like to calculate the DSC on
    for folder in os.listdir(dir_preds):

        if folder=='.DS_Store':
            continue
        fraction_ID = folder
        print(fraction_ID)
        fname = fraction_ID + ".nii"

        #Select organ filepath based on index
        if index==1:
            '''fpath_pred = os.path.join(dir_preds, fraction_ID, 'Prostate_pred.nii')
            fpath_truth = os.path.join(dir_truth, 'ProstateOnly', fname)'''
            fpath_pred = os.path.join(dir_preds, fraction_ID, 'SVs_pred.nii')
            fpath_truth = os.path.join(dir_truth, fname)

        elif index==2:
            fpath_pred = os.path.join(dir_preds, fraction_ID, 'Bladder_pred.nii')
            fpath_truth = os.path.join(dir_truth, 'Bladder', fname)
        elif index==3:
            fpath_pred =os.path.join(dir_preds, fraction_ID, 'Rectum_pred.nii')
            fpath_truth = os.path.join(dir_truth, 'Rectum', fname)


        arr1_sitk = sitk.ReadImage(fpath_pred)
        im1_arr = sitk.GetArrayFromImage(arr1_sitk)

        arr2_sitk = sitk.ReadImage(fpath_truth)
        im2_arr = sitk.GetArrayFromImage(arr2_sitk)

        DSC_indiv = DSC_calc(im1_arr, im2_arr)
        DSC.append(DSC_indiv)

        #Round
        DSC_rounded = [round(elem, 3) for elem in DSC]

    return DSC_rounded





dir_preds = '/Users/sblackledge/Documents/MRL_segmentation_predictions/SVs/attempt4/niftis_validation'
dir_truth = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D/SVsOnly'
index = 1
DSC = export_DSCs(index, dir_truth, dir_preds)
print(DSC)