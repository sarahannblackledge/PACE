import numpy as np
import os
import SimpleITK as sitk

#Create separate dictionaries for PACE and PRISM plans within user-specified directory with id number and filepath as key-value pairs
#inputs:
    #1. dose_dir: string - full file path of directory where mha files are stored

#Output:
    #1. PACE_dict: keys = image id number, value = [fpath of mha file]
    #2. PRISM_dict: keys = image id number, value = [fpath of mha file]

def metadata_from_dose_dir(dose_dir):
    mha_files = np.array([os.path.join(dose_dir, fl) for fl in os.listdir(dose_dir) if "mha" in fl])
    PACE_dict = {}
    PRISM_dict = {}

    for file in mha_files:
    # Extract image id_number to which dose corresponds to
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        im_id = reader.GetMetaData('OnImage')
        plan = reader.GetMetaData('Plan')


        #Populate dictionary: key = image id number, value = fpath of corresponding mha file
        if 'PACE' in plan:
            PACE_dict[im_id] = [file]
        elif 'PRISM' in plan:
            PRISM_dict[im_id] = [file]
    return PACE_dict, PRISM_dict

PACE_dict, PRISM_dict = metadata_from_dose_dir(dose_dir)


