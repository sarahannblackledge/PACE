import SimpleITK as sitk
import numpy as np
from skimage.draw import polygon
import sys
from tqdm import tqdm
import copy
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from dicom_utilities import copy_dicom_tags


def extract_struct_info(rtstruct_dicom):

    # create dictionary containing all structures within rtstruct
    structure_sets_dict = {int(d[0x30060022].value): d for d in rtstruct_dicom[0x30060020]}

    contour_sequences = rtstruct_dicom.ROIContourSequence

    # Create list of contour names
    contour_names = []
    for item in contour_sequences:
        structure_idx = item[0x30060084].value #Tag = 'Referenced ROI number'
        contour_name = structure_sets_dict[structure_idx][0x30060026].value
        contour_names.append(contour_name)

    return structure_sets_dict, contour_names, contour_sequences

def check_mask_list(masks_of_interest, contour_names):
    indices = 0
    name_matches = []
    for m in masks_of_interest:
        if m in contour_names:
            indices = indices + 1
            name_matches.append(m)
        else:
            print(f"Warning: {m} does not exist in structure set")
            print(f" Structures that do exist: {contour_names}")

    if indices < 1:
        print('exiting code: No contours in structure set matching masks_of_interest')
        return []
    else:
        print(f"Contours to be converted: {name_matches}")

    return name_matches

def contour_to_mask(contourSequence, roi_mask, z_locs, orX, orY, spX, spY):
    # For each slice comprising stucture
    for j in contourSequence:
        xyz = j.ContourData
        x = xyz[0::3]
        y = xyz[1::3]
        z = xyz[2::3]
        z_diff = np.abs(z_locs - z[0])
        z_idx = np.where(z_diff == np.min(z_diff))[0][0]
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        x = (x_arr - orX) / spX
        y = (y_arr - orY) / spY
        mask = roi_mask[:, :, z_idx]
        mask_new = np.zeros_like(mask)
        rr, cc = polygon(x, y, mask.shape)
        mask_new[rr, cc] = True
        mask = np.logical_xor(mask, mask_new)
        roi_mask[:, :, z_idx] = mask
    return roi_mask

def make_mask_arr(refim_sitk, masks_of_interest, contour_sequences, structure_sets_dict):
    '''Get 4D array of desired 3D mask arrays.
    inputs:
        1. refim_sitk - sitk image: reference image that masks should correspond to
        2. masks_of_interest - list of strings: Desired structures in RTSTRUCT to convert to a mask
        3. contour sequences - Sequence containing info for each structure in RTSTRUCT
        (extrated from metadata tag in RTSTRUCT).
        4. structure_sets_dict - dict where the key is the Referenced ROI number, and the value is contour name as
        specified in the RTSTRUCT metadata.

    output:
        1. masks - 4D ndarray of size [slices, rows, cols, n_masks].
        2. names - dict: key = index in which contour_name was read. Value = RTSTRUCT contour_name. Only contains names
        if specified in masks_of_interest. If structure does not exist in RTSTRUCT but does exist in masks_of_interest,
        it will not be in the 'names' dictionary.'''
    # Define grid on which masks should be created
    orX, orY, orZ = refim_sitk.GetOrigin()
    szX, szY, szZ = refim_sitk.GetSize()
    spX, spY, spZ = refim_sitk.GetSpacing()
    z_locs = orZ + np.arange(szZ) * spZ

    # Initialize mask and names list/dict for storing individual structures
    masks = []
    names = {}
    mask_idx = 0

    # For each contour itemized in ROIContourSequence tag
    for item in contour_sequences:
        structure_idx = item[0x30060084].value
        contour_name = structure_sets_dict[structure_idx][0x30060026].value

        if contour_name in masks_of_interest:
            roi_mask = np.zeros(refim_sitk.GetSize(), dtype="int")
            contourSequence = item.ContourSequence
            # Manually correct to remove underscore from penile_bulb (if exists)
            if contour_name == 'Penile_Bulb':
                names[mask_idx] = 'PenileBulb'
            else:
                names[mask_idx] = contour_name

            roi_mask = contour_to_mask(contourSequence, roi_mask, z_locs, orX, orY, spX, spY)

            masks.append(roi_mask)
            mask_idx += 1

    masks = np.array(masks).transpose((3, 2, 1, 0))
    return masks, names

def make_dummy_bowel(masks, refim_sitk, ref_series_uid):
    '''There are cases where no bowel was contoured because it was far from the PTV. It is sometimes necessary to create
    empty bowel masks in these cases.

    inputs:
        1. masks - ndarray: 4D array of masks [slices, rows, cols, n_masks]. Must have at least one non-bowel mask to
        use as template
        2. refim_sitk - sitk image: reference image template to ensure rtstruct is created on same grid as corresponding
        image.
        3. ref_series_uid - UID: series UID of rtstruct.

    outputs:
        1. bowel_sitk - sitk image: contains array of zeros (based on image template) with ContourName set as bowel.'''

    #Get first mask in arr
    mask_sitk_model = sitk.GetImageFromArray(masks[:, :, :, 0].astype("uint8"))

    mask_bowel = copy.deepcopy(mask_sitk_model)
    mask_template = sitk.GetArrayFromImage(mask_bowel)
    mask_template.fill(0)
    bowel_sitk = sitk.GetImageFromArray(mask_template.astype("uint8"))
    bowel_sitk.CopyInformation(refim_sitk)
    bowel_sitk.SetMetaData("ContourName", "Bowel")
    bowel_sitk.SetMetaData("CTSeriesUID", ref_series_uid)

    return bowel_sitk


def create_rtstruct_masks(rtstruct_dicom, refim_sitk, masks_of_interest):
    """ Convert rtstruct dicom file to sitk images

    Args
    ====
    rtstruct_dicom : FileDataset
    meta-data from rtstruct dicom file (e.g. as read from pydicom)

    refim_sitk : SimpleITK.Image
    The CT image on which the RTStruct is defined.

    masks_of_interest: list
    Strings indicating which structures should be written to masks



    Return
    ======
    masks : list
    A list of SimpleITK.Image instances for the masks

    """
    #Extract relevant info from RTSTRUCT file
    structure_sets_dict, contour_names, contour_sequences = extract_struct_info(rtstruct_dicom)

    # Check which masks_of_interest exist in RTSTRUCT.
    name_matches = check_mask_list(masks_of_interest, contour_names)

    #Make 4D array containing all 3D masks desired
    masks, names = make_mask_arr(refim_sitk, masks_of_interest, contour_sequences, structure_sets_dict)

    #Check whether 'Bowel' exists in contour set
    TF = 'Bowel' in names.values()

    #Initialize list to store mask sitk image objects
    mask_images = []
    ref_series_uid = rtstruct_dicom[0x3006, 0x10][0][0x3006, 0x12][0][0x3006, 0x14][0][0x20, 0xe].value

    for idx in tqdm(names, leave=False, desc="Creating individual"):
        mask_image_sub = sitk.GetImageFromArray(masks[:, :, :, idx].astype("uint8"))
        mask_image_sub.CopyInformation(refim_sitk)
        mask_image_sub.SetMetaData("ContourName", names[idx])
        mask_image_sub.SetMetaData("CTSeriesUID", ref_series_uid)

        #copy dicom tags
        copy_dicom_tags(mask_image_sub, rtstruct_dicom, ignore_private=True, ignore_groups=())

        if names[idx] in masks_of_interest:
            mask_images.append(mask_image_sub)

    if 'Bowel' in masks_of_interest:
        #If bowel does not exist in contour set:
        if not TF:
            print('Dummy bowel mask created (all zeros)')
            bowel_sitk = make_dummy_bowel(masks, refim_sitk, ref_series_uid)
            mask_images.append(bowel_sitk)

    return mask_images




