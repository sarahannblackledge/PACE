import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from metadata_from_dose_dir import metadata_from_dose_dir


#adjust per image-dose pair
patient = 'NIHR_1'
ids = [97, 1]
plan = 'PACE'

for id in ids:

    #hard-coded variables (consistent in Proknow database directory)
    id_str = 'MR' + str(id)
    dcm_dir = '/Users/sblackledge/Documents/ProKnow_database/' + patient + '/' + id_str
    id_num = str(id)
    if not os.path.isdir(dcm_dir):
        id_str = 'RefMR' + str(id)
        dcm_dir = '/Users/sblackledge/Documents/ProKnow_database/' + patient + '/' + id_str
        id_num = id_str
    im_str = 'MR1'
    dose_dir = '/Users/sblackledge/Documents/ProKnow_database/' + patient + '/' + '/dose'

    #Generate sitk image object from dicom files in specified directory
    files = np.array([os.path.join(dcm_dir, fl) for fl in os.listdir(dcm_dir) if "dcm" in fl and im_str in fl])
    dicoms = np.array([dicom.read_file(fl, stop_before_pixels = True) for fl in files])
    locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
    files = files[np.argsort(locations)]
    mri = sitk.ReadImage(files)

    #Find corresponding PACE dose from 'dose' folder (if exists)
    PACE_dict, PRISM_dict = metadata_from_dose_dir(dose_dir)

    if plan == 'PACE':
        fpath_mha = PACE_dict.get(id_num)
        fraction_num = 5
    elif plan == 'PRISM':
        fpath_mha = PRISM_dict.get(id_num)
        fraction_num = 20

    if fpath_mha is None:
        string_in_string = "No dose corresponding to image {}".format(id_str)
        print(string_in_string)
        continue
    fpath_mha = fpath_mha[0]
    print(fpath_mha)

    #Read in mha file
    dose = sitk.ReadImage(fpath_mha)

    #Spacing
    spX, spY, spZ = dose.GetSpacing()
    dose.SetSpacing((spX * 10, spY * 10, spZ * 10))  # convert from cm to mm
    #Origin
    orX, orY, orZ = dose.GetOrigin()
    dose.SetOrigin((orX * 10, orY * 10, orZ * 10))  # convert from cm to mm

    #Resample dose to corresponding mri
    dose_resample = sitk.Resample(dose, mri, sitk.AffineTransform(3), sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    #Multiply dose by 5 (for PACE) or 20 (for PRISM) to convert between dose per fraction to total dose.
    fraction_dose = sitk.GetArrayFromImage(dose_resample)
    total_dose = fraction_dose*fraction_num
    total_dose_resample = sitk.GetImageFromArray(total_dose)
    total_dose_resample.CopyInformation(dose_resample)

    #Write resampled dose to nifti
    savepath = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/dose'
    fname = patient + '_' + id_str + '.nii'
    savename = os.path.join(savepath, fname)
    sitk.WriteImage(total_dose_resample, savename)

    #Convert mri dicoms into nifti and save
    savepath_mri = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/MRI'
    savename_mri = os.path.join(savepath_mri, fname)
    sitk.WriteImage(mri, savename_mri)


    # Display
    '''dose_resample_arr = sitk.GetArrayFromImage(dose_resample)
    mri_arr = sitk.GetArrayFromImage(mri)

    max_slice = np.where(dose_resample_arr == np.max(dose_resample_arr))[0][0]
    max_dose = np.max(dose_resample_arr)
    print(max_dose)
    plt.imshow(mri_arr[max_slice], cmap='gray')
    plt.contour(dose_resample_arr[max_slice], np.array([0.2, 0.4, 0.6, 0.8]) * max_dose, colors=['g', 'y', 'm', 'r'])
    plt.show()
    print(np.max(dose_resample_arr))'''


