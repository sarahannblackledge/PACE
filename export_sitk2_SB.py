import os
import numpy as np
import SimpleITK as sitk
import pydicom as dicom
import sys

sys.path.append('/Users/sblackledge/PycharmProjects/pythonProject/GENIUSII')
from copy_dicom_tags import copy_dicom_tags
from create_rtstruct_masks_SB import create_rtstruct_masks

'''Saves dicom export from monaco into nifti files. Assumes each fraction is saved in it's own directory and contains only
one RTSTRUCT, RTDOSE, and Session MR. This code is similar to export_sitk_SB, but adapted to the file structure expected
from the Monaco export (Sophie's data). 

Inputs:
    1. export_directory: str - full filepath where data exported from Monaco is stored. Each fraction is exported as an 
    individual folder which contains the MR images, RTPLAN, RTDOSE, and RTSTRUCT. 
    Example: export_directory =  '/Users/sblackledge/Documents/ProKnow_database/Sophie_dataset/patient1/PER022 MR Session 1'
    2. save_dir: str - full filepath where nifti files should be saved. 
    Example: save_dir = '/Users/sblackledge/Documents/GENIUSII_exports/nifti_dump'
    3. patient_name: str - string indicating name of patient. Example: 'g02'

Output:
    Nifti file for every (1) dcm image dataset, (2) relevant structure from the RTSTRUCT.dcm file, and (3) dose cube.
    
'''


def DICOMRawData_to_nifti(export_directory, save_dir, patient_name, masks_of_interest):
    study_uids_blacklist = {}
    floc_el = 0x19100c  # Used to store the file location in read dicoms

    # Create 'images' sub-directory.
    im_dir = os.path.join(save_dir, 'images')
    CHECK_FOLDER = os.path.isdir(im_dir)
    if not CHECK_FOLDER:
        os.makedirs(im_dir)

    # Create 'masks' sub-directory
    mask_dir = os.path.join(save_dir, 'masks3D')
    CHECK_FOLDER = os.path.isdir(mask_dir)
    if not CHECK_FOLDER:
        os.makedirs(mask_dir)

    # Create 'dose' sub-directory
    dose_dir = os.path.join(save_dir, 'dose')
    CHECK_FOLDER = os.path.isdir(dose_dir)
    if not CHECK_FOLDER:
        os.makedirs(dose_dir)

    #Define patient name
    fname = patient_name + '.nii'

    # Load in the mr dicoms, RTSTRUCT, and RTDOSE
    mr_dicoms = {}

    for dicom_file in os.listdir(export_directory):
        if dicom_file == ".DS_Store":
            continue

        try:
            dicom_path = os.path.join(export_directory, dicom_file)
            dcm = dicom.read_file(dicom_path, stop_before_pixels=True)
            series_uid = dcm.SeriesInstanceUID
            modality = dcm.Modality

            if dcm.StudyInstanceUID in study_uids_blacklist.keys():
                break

            # Look for MR images
            if modality == 'MR':
                if not series_uid in mr_dicoms:
                    mr_dicoms[series_uid] = []
                dcm.add_new(floc_el, "ST", dicom_path)
                mr_dicoms[series_uid].append(dcm)

            #Identify rtstrcut
            if modality == 'RTSTRUCT': #Should only be one RS file in directory.
                ref_rtstruct = dicom.read_file(dicom_path, stop_before_pixels=True)
                ref_rtstruct.add_new(floc_el, "ST", dicom_path)
                ref_rtstruct_uid = ref_rtstruct.SOPInstanceUID

            #Identify dose cube
            if modality == 'RTDOSE':
                dose_dcm = dicom.read_file(dicom_path, stop_before_pixels=True)
                dose_dcm.add_new(floc_el, "ST", dicom_path)

        except:
            raise

    # Now organise files in MR lists by ascending slice location
    for series_uid in mr_dicoms:
        slice_locations = [float(dcm.ImagePositionPatient[-1]) for dcm in mr_dicoms[series_uid]]
        mr_dicoms[series_uid] = np.array(mr_dicoms[series_uid])[np.argsort(slice_locations)].tolist()

    # Find the MR image corresponding to the RTSTRUCT and save to nifti
    ref_mr_series_uid = ref_rtstruct[0x3006, 0x10][0][0x3006, 0x12][0][0x3006, 0x14][0][0x20, 0xe].value
    for series_uid in mr_dicoms:
        if series_uid == ref_mr_series_uid:
            ref_ct_study = mr_dicoms[series_uid]
            #study_date = ref_ct_study[0].ContentDate #extract date from first file
            ref_mr_image = sitk.ReadImage([dcm[floc_el].value for dcm in ref_ct_study])  # sitk object for ref CT
            copy_dicom_tags(ref_mr_image, ref_ct_study[0], ignore_private=True)
            #ref_mr_image.SetMetaData('0008,0020', study_date)
            ref_mr_image.SetMetaData('0008,103e', 'MR')

            # Save MR to images sub-directory in 'nifti dump' folder
            save_path = os.path.join(im_dir, fname)
            sitk.WriteImage(ref_mr_image, save_path, True)

            if ref_ct_study is None:
                print("Could not find a CT series corresponding to RTSTRUCT: %s" % str(ref_rtstruct_uid))
                continue

    #Save the dose cube
    ref_dose = sitk.ReadImage([dose_dcm[floc_el].value])
    fpath_dose = os.path.join(dose_dir, fname)
    sitk.WriteImage(ref_dose, fpath_dose, True)

    # Generate masks of each structure in RTSTRUCT.
    rtstruct_images_sub = create_rtstruct_masks(ref_rtstruct, ref_mr_image)  # output: list of sitk objects

    for im in rtstruct_images_sub:
        structure_name = im.GetMetaData("ContourName")
        fpath_structure = os.path.join(mask_dir, structure_name, fname)
        sitk.WriteImage(im, fpath_structure, True)

    return mr_dicoms, ref_mr_image, rtstruct_images_sub


patient_name = 'PER022_tx1'
export_directory = '/Users/sblackledge/Documents/ProKnow_database/Sophie_dataset/patient1/PER022 MR Session 1'
save_dir = '/Users/sblackledge/Documents/ProKnow_database/Sophie_dataset/nifti_dump'
mr_dicoms, ref_mr_image, rtstruct_images_sub = DICOMRawData_to_nifti(export_directory, save_dir, patient_name)

#For reference (hard-coded in create_rtstruct_masks_SB)
masks_of_interest = ['Prostate', 'SeminalVes', 'Rectum', 'Bladder', 'Bowel']

