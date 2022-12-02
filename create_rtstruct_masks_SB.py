import SimpleITK as sitk
import numpy as np
from skimage.draw import polygon
import sys
from tqdm import tqdm
import copy
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from get_python_tags import get_dicom_tags


def create_rtstruct_masks(rtstruct_dicom, ct_image):
    """ Convert rtstruct dicom file to sitk images

    Args
    ====
    fpath_rtstruct : full file path to RTSTRUCT.dcm file

    ct_image : SimpleITK.Image
    The CT image on which the RTStruct is defined.

    save_dir : full file path to directory where masks should be saved

    Return
    ======
    masks : list
    A list of SimpleITK.Image instances for the masks

    """
    # Provides the names
    #0x30060022 = ROI Number
    #0x30060026 = ROI Name
    structure_sets = {int(d[0x30060022].value):d for d in rtstruct_dicom[0x30060020]}

    # masks_of_interest = ['CTVpsv_4000', 'CTVsv', 'PTVpsv_3625', 'PTVsv_3000', 'Bladder', 'Rectum', 'Bowel', 'Prostate', 'PenileBulb', 'SeminalVes']
    # masks_of_interest = ['Rectum', 'Bowel', 'Bladder', 'Penile_Bulb', 'ProstateOnly', 'SVsOnly', 'CTV_Prostate', 'CTV_SVs', 'CTV_Prostate+SVs', 'PTV_4860']
    #masks_of_interest = ['ProstateOnly', 'SVsOnly', 'Rectum', 'Bladder', 'Bowel']
    masks_of_interest = [' PTVpsv_3625', 'CTV_Prostate+SVs']

    orX, orY, orZ = ct_image.GetOrigin()
    szX, szY, szZ = ct_image.GetSize()
    spX, spY, spZ = ct_image.GetSpacing()
    z_locs = orZ + np.arange(szZ) * spZ

    masks = []
    names = {}

    mask_idx = 0
    contour_sequences = rtstruct_dicom.ROIContourSequence

    #Check to see whether desired contour(s) exist
    contour_names = []
    for item in contour_sequences:
        structure_idx = item[0x30060084].value
        contour_name = structure_sets[structure_idx][0x30060026].value
        contour_names.append(contour_name)

    for m in masks_of_interest:
        indices = 0
        if m in contour_names:
            print(f"{m} exists in structure set")
            indices = indices + 1
        else:
            print(f"{m} does not exist in structure set")

    if indices < 1:
        print('exiting code')

        return []

    # For each contour itemized in ROIContourSequence tag
    for item in contour_sequences:
        roi_mask = np.zeros(ct_image.GetSize(), dtype="int")
        contourSequence = item.ContourSequence
        structure_idx = item[0x30060084].value
        contour_name = structure_sets[structure_idx][0x30060026].value

        if contour_name in masks_of_interest:
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
    name_idx_dict = dict((v, k) for k, v in names.items())

    #Convert individual masks into single multiclass mask where each structure is assigned a number from 1 - 5 (background = 0)
    '''example_mask = masks[:, :, :, 0]
    multiclass_mask = np.zeros(example_mask.shape)
    counter = 0
    for i in range(0, np.shape(masks)[-1]):
        indiv_mask = masks[:, :, :, i]
        counter = counter + 1
        indiv_mask = indiv_mask*counter
        multiclass_mask = indiv_mask + multiclass_mask'''

    #Get prostate and SVs and combine into a single structure to account for frequent overlap of contours
    '''
    idx_prostate = name_idx_dict.get('ProstateOnly')
    idx_SVs = name_idx_dict.get('SVsOnly')

    prostate_mask = masks[:, :, :, idx_prostate]
    SV_mask = masks[:, :, :, idx_SVs]
    prostate_U_SVs = prostate_mask + SV_mask
    prostate_U_SVs[prostate_U_SVs > 0] = 1'''

    # Convert individual masks into single 4D multiclass mask where each structure is indicated by the index in the fourth dimension
    '''idx_prostate = name_idx_dict.get('ProstateOnly')
    idx_SVs = name_idx_dict.get('SVsOnly')
    idx_bladder = name_idx_dict.get('Bladder')
    idx_rectum = name_idx_dict.get('Rectum')

    prostate_mask = masks[:,:, :, idx_prostate]
    SV_mask = masks[:, :, :, idx_SVs]
    bladder_mask = masks[:, :, :, idx_bladder]
    rectum_mask = masks[:, :, :, idx_rectum]

    multiclass_mask = np.concatenate((bladder_mask[..., np.newaxis], rectum_mask[..., np.newaxis], prostate_mask[..., np.newaxis], SV_mask[..., np.newaxis]), axis=3)'''

    #generate sitk image for each mask and concatenate into single array
    TF = 'Bowel' in names.values()
    for idx in tqdm(names, leave=False, desc="Creating individual"):
        print(idx)
        mask_image_sub = sitk.GetImageFromArray(masks[:, :, :, idx].astype("uint8"))
        mask_image_sub.CopyInformation(ct_image)
        mask_image_sub.SetMetaData("ContourName", names[idx])
        ref_ct_series_uid = rtstruct_dicom[0x3006, 0x10][0][0x3006, 0x12][0][0x3006, 0x14][0][0x20, 0xe].value
        mask_image_sub.SetMetaData("CTSeriesUID", ref_ct_series_uid)
        for key in tags:
            mask_image_sub.SetMetaData(key, tags[key])
        if names[idx] in masks_of_interest:
            mask_images.append(mask_image_sub)
        #If a bowel contour does not exist, create one, but filled with zeros.
        if not TF:
            mask_bowel = copy.deepcopy(mask_image_sub)
            mask_template = sitk.GetArrayFromImage(mask_bowel)
            mask_template.fill(0)
            bowel_sitk = sitk.GetImageFromArray(mask_template.astype("uint8"))
            bowel_sitk.CopyInformation(ct_image)
            bowel_sitk.SetMetaData("ContourName", "Bowel")
            bowel_sitk.SetMetaData("CTSeriesUID", ref_ct_series_uid)
            mask_images.append(bowel_sitk)

    return mask_images




