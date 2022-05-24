import numpy as np
import os
import SimpleITK as sitk

#Create dictionary indicating image id and plan corresponding to each filename for all mha files in specified directory
#inputs:
    #1. dose_dir: string - full file path of directory where mha files are stored

#Output:
    #1. python dictionary: keys = filepath of mha files, values = [image id number, plan]

def metadata_from_dose_dir(dose_dir):
    files = np.array([os.path.join(dose_dir, fl) for fl in os.listdir(dose_dir) if "mha" in fl])
    id_dict = {}
    for file in files:
    # Extract image id_number to which dose corresponds to
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        im_id = reader.GetMetaData('OnImage')
        plan = reader.GetMetaData('Plan')

        #Populate dictionary: key = filename, value = image id number
        id_dict[file] = [im_id, plan]
    return id_dict



