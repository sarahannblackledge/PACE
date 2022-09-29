import shutil
import os


organ = 'Rectum'
split = 'test'
str = 'niftis_' + split

save_dest = '/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_full_TVTsets/test/rectum_labels_UNET'
prediction_dir = '/Users/sblackledge/Documents/MRL_segmentation_predictions/bladder_prostate_rectum/attempt7_used/' + str
for folder in os.listdir(prediction_dir):
    if 'NIHR' in folder:
        temp_dir = os.path.join(prediction_dir, folder)
        source_file = os.path.join(temp_dir, organ + '_pred.nii')
        patient_name = folder
        dest_file = os.path.join(save_dest, patient_name + '.nii')
        shutil.copyfile(source_file, dest_file)


