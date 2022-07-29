import numpy as np
import pydicom as dicom
import SimpleITK as sitk
import os


def sitk_im_create_simple(im_str, dcm_dir):
    files = np.array([os.path.join(dcm_dir, fl) for fl in os.listdir(dcm_dir) if "dcm" in fl and im_str in fl])
    dicoms = np.array([dicom.read_file(fl, stop_before_pixels = True) for fl in files])
    locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
    files = files[np.argsort(locations)]
    sitk_im = sitk.ReadImage(files)

    return sitk_im

#im_str = 'MR1'
#dicom_dir = '/Users/sblackledge/Documents/ProKnow_database/NIHR_2/MR6'
#ct_image = sitk_im_create_simple(im_str, dicom_dir)