import numpy as np
import os
import SimpleITK as sitk

#Create separate dictionaries for PACE and PRISM plans within user-specified directory with filepath and relevant metadata as key-value pairs
#inputs:
    #1. dose_dir: string - full file path of directory where mha files are stored

#Output:
    #1. PACE_dict: keys = filepath of mha files corresponding to PACE plans, values = [image id number, plan]
    #2. PRISM_dict: keys = filepath of mha files corresponding to PRISM plans, values = [image_id number, plan]

def metadata_from_dose_dir(dose_dir):
    files = np.array([os.path.join(dose_dir, fl) for fl in os.listdir(dose_dir) if "mha" in fl])
    PACE_dict = {}
    PRISM_dict = {}
    for file in files:
    # Extract image id_number to which dose corresponds to
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        im_id = reader.GetMetaData('OnImage')
        plan = reader.GetMetaData('Plan')

        #Populate dictionary: key = filename, value = image id number
        if 'PACE' in plan:
            PACE_dict[file] = [im_id, plan]
        elif 'PRISM' in plan:
            PRISM_dict[file] = [im_id, plan]
    return PACE_dict, PRISM_dict

dose_dir = '/Users/sblackledge/Documents/ProKnow_database/NIHR_1/dose'

PACE_dict, PRISM_dict = metadata_from_dose_dir(dose_dir)

