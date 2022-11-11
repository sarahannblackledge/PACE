import pydicom
import SimpleITK as sitk
import os
from pydicom.uid import generate_uid
import datetime

def convertNsave(arr, ipp, series_id, save_dir, index, fpath_template):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put
    the name of each slice while using a for loop to convert all the slices
    """
    #Generate new uid
    new_SOP_id = generate_uid()

    dicom_file = pydicom.dcmread(fpath_template)

    #Time info
    #dt = datetime.datetime.now()
    #timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    #dicom_file.InstanceCreationTime = timeStr

    dicom_file.SOPInstanceUID = new_SOP_id
    dicom_file.SeriesInstanceUID = series_id

    #dicom_file.MediaStorageSOPInstanceUID = new_SOP_id
    arr = arr.astype('uint16')
    #dicom_file.Rows = arr.shape[0]
    #dicom_file.Columns = arr.shape[1]
    dicom_file.PixelData = arr.tobytes()
    dicom_file.ImagePositionPatient[0] = ipp[0]
    dicom_file.ImagePositionPatient[1] = ipp[1]
    dicom_file.ImagePositionPatient[2] = ipp[2]
    dicom_file.SliceLocation = ipp[2]
    dicom_file.SliceThickness = 1
    dicom_file.InstanceNumber = index + 1
    #dicom_file.PatientOrientation = ['L', 'R']

    dicom_file.save_as(os.path.join(save_dir, f'slice{index}.dcm'))

def nifti_to_dicoms(fpath_nifti_mask, save_dir, fpath_template):

    im_sitk = sitk.ReadImage(fpath_nifti_mask)
    im_arr = sitk.GetArrayFromImage(im_sitk)
    series_id = generate_uid()

    for i in range(im_arr.shape[0]):
        arr = im_arr[i, :, :]
        ipp = im_sitk.TransformIndexToPhysicalPoint((0, 0, i))
        convertNsave(arr, ipp, series_id, save_dir, i, fpath_template)


save_dir = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/dicom_masks/NIHR_1_MR11/prostate'
fpath_nifti_mask = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D/ProstateOnly/NIHR_1_MR11.nii'
fpath_template = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/MR_dicoms_organized/NIHR_1_MR37/MR.1.2.826.0.1.3680043.8.498.82444367742567962188299849025424305321'
#fpath_template = '/Users/sblackledge/Downloads/NIHR-1-MR37/image.0001.dcm'


nifti_to_dicoms(fpath_nifti_mask, save_dir, fpath_template)
