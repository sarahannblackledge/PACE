import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import os

#generates sitk image object from dicom files
#inputs:
    #1. im_str: string - pattern found in all desired dicoms, e.g. '_MR1'.
#    Used to distinguish between other files in directory, such as RTSTRUCTS, REG, etc.

    #2. dcm_dir: string - full file path of directory where dicom images are stored

def sitk_im_create(im_str, dcm_dir):

    # create list of dicom files ensuring that they are ordered based on slice position (ImagePositionPatient)
    files_im = np.array([os.path.join(dcm_dir, fl) for fl in os.listdir(dcm_dir) if "dcm" in fl and im_str in fl])
    series_orig = list(files_im)

    # Preallocate empty arrays
    files_im2 = []
    locations = np.empty([1,1])
    for fl in files_im:
        dicom_i = dicom.read_file(fl, stop_before_pixels = True)
        sd = dicom_i.SeriesDescription
        if 'T2' in sd:
            locs = np.array([float(dicom_i.ImagePositionPatient[-1])])
            locs = np.reshape(locs, (1,1))
            locations = np.concatenate((locations, locs), axis = 0)
            files_im2.append(fl)
    locations = locations[1:,0]
    files_im2 = np.asarray(files_im2)
    files_im2 = files_im2[np.argsort(locations)]
    files_im2 = np.squeeze(files_im2)
    series = files_im2.tolist()

    # Create sitk image object
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series)
    sitk_image = reader.Execute()
    return sitk_image

'''im_str = '_MR_MR'
dcm_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data/0000001/1/Prostate/1PAC/Fraction_1_ATS/DeliveredPlan'
test_sitk_image = sitk_im_create(im_str, dcm_dir)'''
