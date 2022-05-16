import pydicom as dicom
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from skimage.draw import polygon
import sys
import os
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from get_python_tags import get_dicom_tags

def create_rtstruct_mask(fpath_rtstruct, ct_image, save_dir):
    """ Convert rtstruct dicom file to sitk images

    Args
    ====
    fpath_rtstruct : full file path to RTSTRUCT.dcm file

    ct_image : SimpleITK.Image
    The CT image on which the RTStruct is defined.

    Return
    ======
    masks : list
    A list of SimpleITK.Image instances for the masks

    """
    #Get load RTSTRUCT into pydicom dataset
    rtstruct_dicom = dicom.read_file(fpath_rtstruct)

    # Provides the names
    #0x30060022 = ROI Number
    #0x30060026 = ROI Name
    structure_sets = {int(d[0x30060022].value):d for d in rtstruct_dicom[0x30060020]}

    orX, orY, orZ = ct_image.GetOrigin()
    szX, szY, szZ = ct_image.GetSize()
    spX, spY, spZ = ct_image.GetSpacing()
    z_locs = orZ + np.arange(szZ) * spZ

    masks = []
    names = {}

    mask_idx = 0
    contour_sequences = rtstruct_dicom.ROIContourSequence

    # For each contour itemized in ROIContourSequence tag
    for item in contour_sequences:
        roi_mask = np.zeros(ct_image.GetSize(), dtype="bool")
        contourSequence = item.ContourSequence
        structure_idx = item[0x30060084].value
        contour_name = structure_sets[structure_idx][0x30060026].value
        names[mask_idx] = contour_name


        #For each slice comprising stucture
        for j in contourSequence:
            xyz = j.ContourData
            x = xyz[0::3]
            y = xyz[1::3]
            z = xyz[2::3]
            z_diff = np.abs(z_locs - z[0])
            z_idx = np.where(z_diff == np.min(z_diff))[0][0]
            x_arr  =np.asarray(x)
            y_arr = np.asarray(y)
            x = (x_arr - orX) / spX
            y = (y_arr - orY) / spY
            mask = roi_mask[:, :, z_idx]
            mask_new = np.zeros_like(mask)
            rr, cc = polygon(x, y, mask.shape)
            mask_new[rr, cc] = True
            mask = np.logical_xor(mask, mask_new)
            roi_mask[:, :, z_idx] = mask

        masks.append(roi_mask)
        mask_idx += 1

    masks = np.array(masks).transpose((3, 2, 1, 0))
    mask_images = []
    tags = get_dicom_tags(rtstruct_dicom, ignore_private=True, ignore_groups=[0x3006])
    for idx in tqdm(names, leave=False, desc="Creating individual"):
        mask_image_sub = sitk.GetImageFromArray(masks[:, :, :, idx].astype("uint8"))
        mask_image_sub.CopyInformation(ct_image)
        mask_image_sub.SetMetaData("ContourName", names[idx])
        ref_ct_series_uid = rtstruct_dicom[0x3006, 0x10][0][0x3006, 0x12][0][0x3006, 0x14][0][0x20, 0xe].value
        mask_image_sub.SetMetaData("CTSeriesUID", ref_ct_series_uid)
        #Convert masks into simple ITK images
        for key in tags:
            mask_image_sub.SetMetaData(key, tags[key])
        mask_images.append(mask_image_sub)
        head_tail = os.path.split(fpath_rtstruct)
        fname_mask = head_tail[1][:-4] + '_' + names[idx] + '.nii'
        fpath_mask = os.path.join(save_dir, fname_mask)
        sitk.WriteImage(mask_image_sub, fpath_mask)
    return mask_images



im_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data/0000001/1/Prostate/1PAC/Fraction_1_ATS/DeliveredPlan'
im_str = '_MR_MR'

# Get RTstruct
fpath_rtstruct = im_dir + '/0000001_RTSTRUCT_MR1xT_SS.dcm'


# Get sitk image object for CT corresponding to RTSTRUCT
files_im = np.array([os.path.join(im_dir, fl) for fl in os.listdir(im_dir) if "dcm" in fl and im_str in fl])
dicoms = np.array([dicom.read_file(fl, stop_before_pixels=True) for fl in files_im])
locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
files_im = files_im[np.argsort(locations)]
im3D = sitk.ReadImage(files_im)

#Specify directory where masks should be saved
save_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data/nifti_mask_dump'

#Generate masks from RTstructs
masks_3D = create_rtstruct_mask(fpath_rtstruct, im3D, save_dir)

