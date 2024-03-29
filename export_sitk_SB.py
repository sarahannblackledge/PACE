import numpy as np
import pydicom as dicom
import os
import SimpleITK as sitk
import sys
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from copy_dicom_tags import copy_dicom_tags
from key import key
from create_rtstruct_masks_SB import create_rtstruct_masks

study_uids_blacklist = {}
pat_directory = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE'
floc_el = 0x19100c # Used to store the file location in read dicoms

dose_directory = os.path.join(pat_directory, "RTDOSE")
rtstruct_directory = os.path.join(pat_directory, "RTSTRUCT")
plan_directory = os.path.join(pat_directory, "RTPLAN")
ct_directory = os.path.join(pat_directory, "MR")


# Load in the mr dicoms
ct_dicoms = {}
for ct_date in os.listdir(ct_directory):
    if ct_date == ".DS_Store":
        continue
    date_directory = os.path.join(ct_directory, ct_date)
    for dicom_file in os.listdir(date_directory):
        try:
            dicom_path = os.path.join(date_directory, dicom_file)
            dcm = dicom.read_file(dicom_path, stop_before_pixels=True)
            series_uid = dcm.SeriesInstanceUID
            if dcm.StudyInstanceUID in study_uids_blacklist.keys():
                break
            if not series_uid in ct_dicoms:
                ct_dicoms[series_uid] = []
            dcm.add_new(floc_el, "ST", dicom_path)
            ct_dicoms[series_uid].append(dcm)
        except:
            raise

    # Now organise by ascending slice location
    for series_uid in ct_dicoms:
        slice_locations = [float(dcm.ImagePositionPatient[-1]) for dcm in ct_dicoms[series_uid]]
        ct_dicoms[series_uid] = np.array(ct_dicoms[series_uid])[np.argsort(slice_locations)].tolist()


# Load in the struct dicoms
rtstruct_dicoms = []
for dicom_file in os.listdir(rtstruct_directory):
    if dicom_file == ".DS_Store":
        continue
    try:
        rtstruct_path = os.path.join(rtstruct_directory, dicom_file)
        dcm = dicom.read_file(rtstruct_path)
        if dcm.StudyInstanceUID in study_uids_blacklist.keys():
            continue
        dcm.add_new(floc_el, "ST", rtstruct_path)
        rtstruct_dicoms.append(dcm)
    except:
        raise

# Load in the plan dicoms
plan_dicoms = []
for dicom_file in os.listdir(plan_directory):
    if dicom_file == ".DS_Store":
        continue
    try:
        plan_path = os.path.join(plan_directory, dicom_file)
        dcm = dicom.read_file(plan_path)
        if dcm.StudyInstanceUID in study_uids_blacklist.keys():
            continue
        dcm.add_new(floc_el, "ST", plan_path)
        plan_dicoms.append(dcm)
    except:
        raise

# Load in the dose dicoms
dose_dicoms = []
for dicom_file in os.listdir(dose_directory):
    if dicom_file == ".DS_Store":
        continue
    try:
        dose_path = os.path.join(dose_directory, dicom_file)
        dcm = dicom.read_file(dose_path)
        if dcm.StudyInstanceUID in study_uids_blacklist.keys():
            continue
        if not dcm.DoseSummationType == "PLAN":
            continue # Ignore beams and fractions.
        add_dose = True
        for dose_dicom in dose_dicoms:
            dose_array_new = dcm.pixel_array
            dose_array_old = dose_dicom.pixel_array
            if np.allclose(dose_array_new.shape, dose_array_old.shape):
                if np.abs(np.sum(dose_array_old - dose_array_new)) < 1e-5:
                    add_dose = False
        if add_dose:
            dcm.add_new(floc_el, "ST", dose_path)
            dose_dicoms.append(dcm)
    except:
        raise


for dose_dicom in dose_dicoms:

    try:

        # Find the plan
        ref_plan_uid = dose_dicom[0x300c,0x0002][0][0x0008, 0x1155].value
        ref_plan = None
        for plan in plan_dicoms:
            #print(plan.StudyDescription)
            if plan.SOPInstanceUID == ref_plan_uid:
                ref_plan = plan
        if ref_plan is None:
            print("Could not find a plan for dose cube: %s" % ref_plan_uid)
            continue

        # Find the contour set
        ref_rtstruct_uid = ref_plan[0x300c,0x0060][0][0x0008, 0x1155].value
        ref_rtstruct_uid_str = str(ref_rtstruct_uid)
        ref_rtstruct = None
        for rtstruct in rtstruct_dicoms:
            if rtstruct.SOPInstanceUID == ref_rtstruct_uid:
                ref_rtstruct = rtstruct
        if ref_rtstruct is None:
            print("Could not find a rtstruct for dose cube: %s" % ref_rtstruct_uid_str)
            continue

        # Find the CT images
        ref_ct_series_uid = ref_rtstruct[0x3006, 0x10][0][0x3006,0x12][0][0x3006, 0x14][0][0x20,0xe].value
        ref_ct_series_uid_str = str(ref_ct_series_uid)
        for series_uid in ct_dicoms:
            if series_uid == ref_ct_series_uid:
                ref_ct_study = ct_dicoms[series_uid]
        if ref_ct_study is None:
            print("Could not find a CT series for dose cube: %s" % ref_ct_series_uid_str)
            continue

        #Look up patient id based on database ID number. Hard coded in 'db_id_key.xlsx'

        db_id = ref_plan.StudyDescription
        print(db_id)
        id_arr = key()[:,1]
        index = np.where(id_arr==db_id)[0]
        TF = np.any(index)
        counter = 0
        if TF:
            print('valid db id number found')
            index= index[0]

            names_arr = key()[:,0]
            pat_id = names_arr[index]
        else:
            print('junk')
            counter = counter + 1
            pat_id = 'unknown_%d'%counter
            db_id = str(counter)
            counter = counter + 1

        # Now save!
        output_directory = os.path.join(pat_directory, "nifti_dump")

        path_dose = os.path.join(output_directory, "dose")
        path_masks3D = os.path.join(output_directory, 'masks3D')
        path_mri = os.path.join(output_directory, 'MRI')

        fname = pat_id + "_MR" + db_id + ".nii"

        if not os.path.exists(path_dose):
            os.makedirs(path_dose)

        if not os.path.exists(path_mri):
            os.makedirs(path_mri)

        if not os.path.exists(path_masks3D):
            os.makedirs(path_masks3D)


        # Create the CT image if not already there
        study_path_mri = os.path.join(path_mri, fname)
        if os.path.exists(study_path_mri):
            ct_image = sitk.ReadImage(study_path_mri)
        else:
            ct_image = sitk.ReadImage([dcm[floc_el].value for dcm in ref_ct_study])
            copy_dicom_tags(ct_image, ref_ct_study[0], ignore_private=True)
            sitk.WriteImage(ct_image, study_path_mri, True)

        # Create the dose cube and resample to the CT image
        study_path_dose = os.path.join(path_dose, fname)
        if os.path.exists(study_path_dose):
            dose_image = sitk.ReadImage(study_path_dose)
        else:
            scaling = dose_dicom[0x3004, 0xe].value
            dose_image = sitk.ReadImage(dose_dicom[floc_el].value)
            dose_image = sitk.Resample(dose_image, ct_image, sitk.AffineTransform(3), sitk.sitkLinear, 0.0, sitk.sitkFloat64)
            dose_image = dose_image * scaling
            copy_dicom_tags(dose_image, dose_dicom, ignore_private=True)
            sitk.WriteImage(dose_image, study_path_dose, True)

        # Create the RTstruct

        try:
            rtstruct_images_sub = create_rtstruct_masks(ref_rtstruct, ct_image)

            for im in rtstruct_images_sub:
                organ = im.GetMetaData("ContourName")
                fpath_organ = os.path.join(path_masks3D, organ)
                if not os.path.exists(fpath_organ):
                    os.makedirs(fpath_organ)

                sitk.WriteImage(im, os.path.join(fpath_organ, fname), True)
        except:
            raise
            print('Something dodgy with your sitk mask list')

    except:
        raise
        print("Could not process: %s - %s", dose_dicom.PatientName, dose_dicom.SOPInstanceUID)



