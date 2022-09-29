import os
import random
import glob
import shutil


def allocate_ids(n_training, n_val):
    random.seed(1)

    #training set
    list_full = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    training_ids = random.sample(list_full, n_training)

    #Validation set
    list_reduced = [x for x in list_full if x not in training_ids]

    val_ids = random.sample(list_reduced, n_val)

    #Test set (remaining values in list)
    test_ids = [x for x in list_reduced if x not in val_ids]

    return training_ids, val_ids, test_ids

def split_data_AI(desired_ids, source_dir, destination_dir):

    #Loop through all patients in directory
    for id in desired_ids:
        file_name = 'NIHR_' + str(id) + '_'
        fullpath = source_dir + '/' + file_name + '*'

        #Loop through every file corresponding to particular patient
        for item in glob.glob(fullpath):
            shutil.copy(item, destination_dir)


n_training = 11
n_val = 2
test_dir_img = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_cropped/test/images'
test_dir_mask = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_cropped/test/bowel_labels'

training_dir_img = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_cropped/training/images'
training_dir_mask = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_full_TVTsets/training/prostate_labels'

val_dir_img = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_cropped/validation/images'
val_dir_mask = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_cropped/validation/bowel_labels/'

source_dir_img = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/MRI'
source_dir_mask = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump/masks3D_no_overlap/ProstateOnly'

training_ids, val_ids, test_ids = allocate_ids(n_training, n_val)

#Populate training folders
#split_data_AI(training_ids, source_dir_img, training_dir_img)
split_data_AI(training_ids, source_dir_mask, training_dir_mask)

#Populate validation folders
#split_data_AI(val_ids, source_dir_img, val_dir_img)
#split_data_AI(val_ids, source_dir_mask, val_dir_mask)

#Populate test folders
#split_data_AI(test_ids, source_dir_img, test_dir_img)
#split_data_AI(test_ids, source_dir_mask, test_dir_mask)

