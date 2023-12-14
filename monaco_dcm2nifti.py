import os
import numpy as np
import SimpleITK as sitk
import pydicom as dicom
import sys
import matplotlib.pyplot as plt

sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from copy_dicom_tags import copy_dicom_tags


'''

Inputs:
    1. ct_directory: str - full filepath where data exported from RayStation are stored. This contains a dump of 
    all CBCT, CT, RTSTRUCT, and REG dicoms, and is not organized in an intuitive way.
    Example: ct_directory =  '/Users/sblackledge/Documents/GENIUSII_exports/RayStation/g02/Raystation_CTdump'
    2. save_dir: str - full filepath where nifti files should be saved. 
    Example: save_dir = '/Users/sblackledge/Documents/GENIUSII_exports/nifti_dump'
    3. patient_name: str - string indicating name of patient. Example: 'g02'
    4. masks_of_interest: list of strings: exact names of masks that should be exported and saved as niftis.
        example: masks_of_interest = ['Bladder', 'PTV45_1', 'PTV45_2', 'PTV45_3', 'PTV45_Robust', 'Rectum', 'CTV-E', 'CTV-T HRinit', 'CTV-T LRinit_1_Full']

Output:
    Nifti file for every (1) dcm image dataset and (2) relevant structure from the RTSTRUCT.dcm file exported from RayStation
    Note: no date or name information is stored in the metadata of these nifti files, so they are considered fully anonymized.
    HOWEVER, the nifti filename contains the study date by default. I recommend changing this manually retrospectively once
    all desired data has been converted to nifti format (e.g. Fraction1.nii).
'''


def DICOMRawData_to_nifti(ct_directory, save_dir, patient_name):
    study_uids_blacklist = {}
    floc_el = 0x19100c  # Used to store the file location in read dicoms
    default_description = 'unknown'

    # Create 'images' sub-directory.
    im_dir = os.path.join(save_dir, 'images')
    CHECK_FOLDER = os.path.isdir(im_dir)
    if not CHECK_FOLDER:
        os.makedirs(im_dir)

    # Create 'masks' sub-directory
    mask_dir = os.path.join(save_dir, 'masks', patient_name)
    CHECK_FOLDER = os.path.isdir(mask_dir)
    if not CHECK_FOLDER:
        os.makedirs(mask_dir)

    # Create 'dose' sub-directory. Note: The dose from each beam angle is saved as a separate file
    dose_dir = os.path.join(save_dir, 'dose')
    CHECK_FOLDER = os.path.isdir(dose_dir)
    if not CHECK_FOLDER:
        os.makedirs(dose_dir)

    #Define patient name
    fname = patient_name + '.nii'

    # Load in the ct dicoms, REG dicoms, and RTSTRUCT dicoms as separate lists
    ct_dicoms = {}
    dose = {}
    scaling = {}

    counter = 0

    for dicom_file in os.listdir(ct_directory):
        if dicom_file == ".DS_Store":
            continue

        try:
            dicom_path = os.path.join(ct_directory, dicom_file)
            dcm = dicom.read_file(dicom_path, stop_before_pixels=True)
            series_uid = dcm.SeriesInstanceUID

            modality = dcm.Modality
            if dcm.StudyInstanceUID in study_uids_blacklist.keys():
                break

            if modality == 'MR':
                if not series_uid in ct_dicoms:
                    ct_dicoms[series_uid] = []
                dcm.add_new(floc_el, "ST", dicom_path)
                if 'SeriesDescription' not in dcm:
                    dcm.add_new('SeriesDescription', "ST", default_description)
                ct_dicoms[series_uid].append(dcm)

            if modality == 'RTSTRUCT':
                rtstruct = dicom.read_file(dicom_path, stop_before_pixels=True)
                rtstruct.add_new(floc_el, "ST", dicom_path)
                ref_id = rtstruct[0x3006, 0x10][0][0x3006, 0x12][0][0x3006, 0x14][0][0x20, 0xe].value

            if modality == 'RTDOSE':
                counter = counter + 1
                print(counter)
                dose_dcm = dicom.read_file(dicom_path, stop_before_pixels=True)
                dose_dcm.add_new(floc_el, "ST", dicom_path)
                dose_grid_scaling = np.float64(dose_dcm.DoseGridScaling)

                dose_by_beam = sitk.ReadImage([dose_dcm[floc_el].value])
                dose[counter] = dose_by_beam
                scaling[counter] = dose_grid_scaling
        except:
            raise

    # Now organise files in CT lists by ascending slice location
    for series_uid in ct_dicoms:
        slice_locations = [float(dcm.ImagePositionPatient[-1]) for dcm in ct_dicoms[series_uid]]
        ct_dicoms[series_uid] = np.array(ct_dicoms[series_uid])[np.argsort(slice_locations)].tolist()


    # Find the MR image corresponding to the RTSTRUCT and save to nifti
    for series_uid in ct_dicoms:
        if series_uid == ref_id:
            ref_ct_study = ct_dicoms[series_uid]
            ref_mr_image = sitk.ReadImage([dcm[floc_el].value for dcm in ref_ct_study])  # sitk object for MR
            copy_dicom_tags(ref_mr_image, ref_ct_study[0], ignore_private=True)
            ref_mr_image.SetMetaData('0008,103e', 'MR')

            # Save MR to images sub-directory in 'nifti dump' folder
            save_path = os.path.join(im_dir, fname)
            sitk.WriteImage(ref_mr_image, save_path, True)

            if ref_ct_study is None:
                print("Could not find a CT series corresponding to RTSTRUCT: %s" % str(ref_id))
                continue

    #Extract relevant parameters from template
    dose_template = sitk.GetArrayFromImage(dose[1])

    #Initialize dose cube
    total_dose = np.zeros(dose_template.shape, dtype='float64')

    #Sum the dose from the individual beam angles
    for i, angle in enumerate(dose):
        dose_sitk = dose[i+1]
        dose_arr = sitk.GetArrayFromImage(dose_sitk)
        dose_arr2 = dose_arr*scaling[i+1]
        total_dose = total_dose + dose_arr2



    #Check; bug where dose is double when computed from individual beam angles; check max dose and divide by 2 if necessary
    max_dose = np.amax(total_dose)
    print('computed max dose is: ' + str(max_dose))
    if max_dose > 55 and i > 1:
        total_dose = total_dose/2
        print('correction factor applied. New max dose is: ' + str(np.amax(total_dose)))

    #Save as nifti
    dose_sitk2 = sitk.GetImageFromArray(total_dose, isVector=False)
    dose_sitk2.CopyInformation(dose[1])
    savepath = os.path.join(dose_dir, fname)
    print(savepath)
    sitk.WriteImage(dose_sitk2, savepath, useCompression=True)


################################################################################
#Bethany dataset example
patient_name = 'PAC23004 CT Ref prop MR8'
ct_directory = os.path.join('/Users/sblackledge/Documents/MRL_plan_database/monaco_dump', patient_name)
save_dir = '/Users/sblackledge/Documents/MRL_plan_database/nifti_dump'
#Run
DICOMRawData_to_nifti(ct_directory, save_dir, patient_name)

