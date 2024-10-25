import os
import re
import shutil

def reformat_NIHR_to_nnUNet(input_str):
    ''' constructs new 'case_identifier' from NIHR naming convention used in Helen's proknow database.
    Original naming convention: NIHR_X_MRYY.
    New naming convention: mrl_XXXYYY

    input:
        1. input_str -str - original NIHR filename (including extension - assumes .nii.gz)
    output:
        2. formatted_str - str - new filename compatible with nnUNet naming convention (including .nii.gz extension)'''
    # Use regex to capture the first number and the number after 'MR'
    match = re.match(r"[^_]+_(\d+)_MR(\d+)", input_str)

    if match:
        # Convert the captured numbers to 3-digit strings
        first_number = int(match.group(1))
        mr_number = int(match.group(2))

        formatted_str = f"MRL_{first_number:03d}{mr_number:03d}.nii.gz"
        return formatted_str
    else:
        raise ValueError("Input string format is incorrect")


def add_channel_identifier(channel_str, filename):
    '''Add channel identifier to string (required for nnUNet, even if only one channel used)
    inputs:
        1. channel_str - str: 4-digit string corresponding to channel (i.e. '0000')
        2. filename - str: original filename (with extension)'''
    # Separate the filename and the full extension
    base_name, ext = os.path.splitext(filename)

    # If the extension is multipart (like .nii.gz), split again
    if ext == ".gz":
        base_name, second_ext = os.path.splitext(base_name)
        ext = second_ext + ext  # Combine for .nii.gz

    # Add '_0000' before the extension
    new_filename = f"{base_name}_{channel_str}{ext}"

    return new_filename


def copy_and_rename(source_dir, dest_dir, channel_str: str = ""):

    for file in os.listdir(source_dir):
        if file.endswith('.nii.gz'):
            nnUNet_name = reformat_NIHR_to_nnUNet(file)
            if channel_str:
                nnUNet_name = add_channel_identifier(channel_str, nnUNet_name)

            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(dest_dir, nnUNet_name)
            shutil.copy(source_path, dest_path)

    print('Copy and rename complete')

#Source data
source_dir_root = '/Users/sblackledge/DATA/ProKnow_database/RMH_proknow/proknowPACE/nifti_dump4'
source_dir_ims = os.path.join(source_dir_root, 'MRI')
source_dir_labels = os.path.join(source_dir_root, 'masks3D_PBR')

#Destination
dest_dir_root = '/Users/sblackledge/Documents/master_MRLprostate_database/nnUNet_raw'
dest_dir_ims = os.path.join(dest_dir_root, 'imagesAll')
dest_dir_labels = os.path.join(dest_dir_root, 'labelsAll')


#Copy MR images
copy_and_rename(source_dir_ims, dest_dir_ims, '0000')
copy_and_rename(source_dir_labels, dest_dir_labels) #Don't add channel identifier to labels