import numpy as np
import SimpleITK as sitk
import pydicom as dicom
import os


def dicom2nifti(im_str, dcm_dir, save_dir):

    # Converts all dicom images with user-specified pattern in filename to nifti series
    # inputs:
    #     1. im_str = str - string  indicating naming pattern to distinguish image files from other potential dicom files (i.e. RTstructs)
    #     2. dcm_dir = str - string specifying directory where dicom images are stored.
    # output:
    #     1. nifti version of dicom files in same directory as dcm_dir.
    # --------------------------------------------------------------------------------------------------------

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
    image = reader.Execute()

    # Save as nifti file
    patient_id = os.path.split(series[0])[1][0:-8]
    fname = patient_id + '.nii'
    fpath = os.path.join(save_dir, fname)
    sitk.WriteImage(image, fpath)

# Hard-code variables consistent across all fractions and patients
save_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data/nifti_dump'
im_str = '_MR_'
parent_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data'

# # TEST file
# im_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data/0000664/1/Prostate/1PAB/Fraction_2_ATS/DeliveredPlan'
# dicom2nifti(im_str,im_dir, save_dir)
# raise

# Loop through all patients in directory
for patient_dir in os.listdir(parent_dir):
     if '000' in patient_dir:
         opt1 = os.path.join(parent_dir, patient_dir, '1/Prostate/1PAC')
         opt2 = os.path.join(parent_dir, patient_dir, '1/Prostate/1PAb')
         if os.path.isdir(opt1):
             fraction_dir = opt1
         elif os.path.isdir(opt2):
             fraction_dir = opt2
         for subdir in os.listdir(fraction_dir):
            if 'Fraction_' in subdir:
                im_dir = os.path.join(fraction_dir, subdir, 'DeliveredPlan')
                dicom2nifti(im_str, im_dir, save_dir)


