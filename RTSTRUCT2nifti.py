import pydicom as dicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

import os
import sys
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from RTSTRUCT2mask import create_rtstruct_mask

im_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data/0000001/1/Prostate/1PAC/Fraction_1_ATS/DeliveredPlan'
im_str = '_MR_MR'

# Get RTstruct
fpath_rtstruct = im_dir + '/0000001_RTSTRUCT_MR1xT_SS.dcm'
rtstruct_dicom = dicom.read_file(fpath_rtstruct)

# Get sitk image object for CT corresponding to RTSTRUCT
files_im = np.array([os.path.join(im_dir, fl) for fl in os.listdir(im_dir) if "dcm" in fl and im_str in fl])
dicoms = np.array([dicom.read_file(fl, stop_before_pixels=True) for fl in files_im])
locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
files_im = files_im[np.argsort(locations)]
im3D = sitk.ReadImage(files_im)

#Generate masks from RTstructs
masks_3D = create_rtstruct_mask(rtstruct_dicom, im3D)