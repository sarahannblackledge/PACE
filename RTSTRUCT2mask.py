import pydicom as dicom
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from skimage.draw import polygon
import sys
import os
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from get_python_tags import get_dicom_tags
from sitk_im_create import sitk_im_create
from sitk_im_create_simple import sitk_im_create_simple

def create_rtstruct_mask(fpath_rtstruct, ct_image, save_dir):
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
    #Get load RTSTRUCT into pydicom dataset
    rtstruct_dicom = dicom.read_file(fpath_rtstruct)

    # Provides the names
    #0x30060022 = ROI Number
    #0x30060026 = ROI Name
    structure_sets = {int(d[0x30060022].value):d for d in rtstruct_dicom[0x30060020]}

    # masks_of_interest = ['CTVpsv_4000', 'CTVsv', 'PTVpsv_3625', 'PTVsv_3000', 'Bladder', 'Rectum', 'Bowel', 'Prostate', 'PenileBulb', 'SeminalVes']
    # masks_of_interest = ['Rectum', 'Bowel', 'Bladder', 'Penile_Bulb', 'ProstateOnly', 'SVsOnly', 'CTV_Prostate', 'CTV_SVs', 'CTV_Prostate+SVs', 'PTV_4860']
    masks_of_interest = ['ProstateOnly', 'SVsOnly', 'Rectum', 'Bowel', 'Bladder', 'Penile_Bulb']

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
        print(contour_name)

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



    for idx in tqdm(names, leave=False, desc="Creating individual"):
        print(idx)
        mask_image_sub = sitk.GetImageFromArray(masks[:, :, :, idx].astype("uint8"))
        mask_image_sub.CopyInformation(ct_image)
        mask_image_sub.SetMetaData("ContourName", names[idx])
        ref_ct_series_uid = rtstruct_dicom[0x3006, 0x10][0][0x3006, 0x12][0][0x3006, 0x14][0][0x20, 0xe].value
        mask_image_sub.SetMetaData("CTSeriesUID", ref_ct_series_uid)
        #Convert masks into simple ITK images
        for key in tags:
            mask_image_sub.SetMetaData(key, tags[key])
        if names[idx] in masks_of_interest:
            mask_images.append(mask_image_sub)

            head_tail = os.path.split(fpath_rtstruct)

            #Naming convention for MRL export data

            '''fname_mask1 = head_tail[1][:-4] + '_' + names[idx] + '.nii'
            #fname_mask2 = head_tail[1][:-4] + '_' + names[idx] + '.nrrd'
            fpath_mask1 = os.path.join(save_dir, fname_mask1)
            #fpath_mask2 = os.path.join('/Users/sblackledge/Documents/MRL_prostate_5FRAC_data/nrrd_mask_dump', fname_mask2)'''

            #Naming convention for ProKnow Database
            n1 = os.path.basename(os.path.normpath(head_tail[0]))
            n2 = os.path.basename(os.path.abspath(os.path.join(head_tail[0], os.pardir)))
            fname = n2 + '_' + n1 + '_' + names[idx] + '.nii'
            fpath_mask1 = os.path.join(save_dir, fname)

            sitk.WriteImage(mask_image_sub, fpath_mask1)
            #sitk.WriteImage(mask_image_sub, fpath_mask2)
    return mask_images, masks


#hard-coded variables

#parent_dir = '/Users/sblackledge/Documents/MRL_prostate_5FRAC_data'

# Loop through all patients in MRL directory
'''for patient_dir in os.listdir(parent_dir):
     if '000' in patient_dir:
         opt1 = os.path.join(parent_dir, patient_dir, '1/Prostate/1PAC')
         opt2 = os.path.join(parent_dir, patient_dir, '1/Prostate/1PAb')
         if os.path.isdir(opt1):
             fraction_dir = opt1
         elif os.path.isdir(opt2):
             fraction_dir = opt2
         for subdir in os.listdir(fraction_dir):
            if 'Fraction_' in subdir:
                dcm_dir = os.path.join(fraction_dir, subdir, 'DeliveredPlan')
                mr_image = sitk_im_create(im_str, dcm_dir)
                fpath_rtstruct = [os.path.join(dcm_dir, fl) for fl in os.listdir(dcm_dir) if
                                  "dcm" in fl and 'RTSTRUCT' in fl]
                fpath_rtstruct = fpath_rtstruct[0]
                create_rtstruct_mask(fpath_rtstruct, mr_image, save_dir)'''




#Generate masks from RTstruct (single fraction)
dcm_dir = '/Users/sblackledge/Documents/ProKnow_database/NIHR_1/MR37'
im_str = 'MR1'
im3D = sitk_im_create_simple(im_str, dcm_dir)
save_dir = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/masks_multiclass'
# Get RTstruct
fpath_rtstruct = dcm_dir + '/RS1.2.752.243.1.1.20210408134521712.7400.20617.dcm'
masks_3D, masks = create_rtstruct_mask(fpath_rtstruct, im3D, save_dir)

#Loop through ProKnow database
'''parent_dir = '/Users/sblackledge/Documents/ProKnow_database'
save_dir = '/Users/sblackledge/Documents/ProKnow_database/nifti_dump_full/mask'
im_str = 'MR1'

for patient_dir in os.listdir(parent_dir):
    if 'NIHR' in patient_dir:
        im_dir = os.path.join(parent_dir, patient_dir)
        for subdir in os.listdir(im_dir):
            if 'MR' in subdir:
                #Create sitk image (MR)
                dcm_dir = os.path.join(im_dir, subdir)
                mr_image = sitk_im_create_simple(im_str, dcm_dir)

                #Extract corresponding RTSTRUCT fpath
                fpath_rtstruct = [os.path.join(dcm_dir, fl) for fl in os.listdir(dcm_dir) if "dcm" in fl and 'RS1' in fl]
                fpath_rtstruct = fpath_rtstruct[0]

                #Save all desired masks as nifti files in save_dir
                create_rtstruct_mask(fpath_rtstruct, mr_image, save_dir)'''







