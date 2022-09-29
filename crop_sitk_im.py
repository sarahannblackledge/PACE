import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np

'''Function to get length, width, and depth (in pixels) of bounding box around specified isodose level
inputs:
    1. fpath_nifti_dose: str - full file path to where nifti dose cubes are stored
    2. dose_level: int - Dose level (in units of Gy) where you would crop
    
output:
    xwidth, ywidth, zwidth: int - number of pixels in bounding box x, y, and z directions '''
def get_cropped_size(fpath_nifti_dose, dose_level):

    sitk_dose = sitk.ReadImage(fpath_nifti_dose)

    #Create mask image that is just non-background pixels
    fg_mask = (sitk_dose > dose_level)

    #Compute shape statistics on the mask
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(fg_mask)

    # Get the bounds of the mask.
    # Bounds are given as [Xstart, Ystart, Zstart, Xwidth, Ywidth, Zwidth]. Units = pixels
    bounds = lsif.GetBoundingBox(1)

    xwidth = bounds[3]
    ywidth = bounds[4]
    zwidth = bounds[5]

    return(xwidth, ywidth, zwidth)

# Plot histograms
def disp_histogram(width_val):
    plt.figure()
    plt.hist(np.asarray(width_val), 10)
    plt.xlabel('width (pixels)')
    plt.ylabel('Frequency')
    plt.show()

'''Loop through all nifti files in directory to get distribution of bounding box sizes in each dimension
    inputs:
        1. nifti_dir: str- full filepath to directory where all dose cube niftis (training set) are stored.
        2. dose_level: int - dose (Gy) over which to find bounding box
    output:
        1. max_x, max_y, max_z: ints - values of maximum dimensions over all niftis in direcory in x, y, and z directions
        respectively. Units = pixels.
        2. bounding_box_dims: np array of ints: Same as output (1), but saved into single np array.'''
def find_max_dims(nifti_dir, dose_level):

    xwidth = []
    ywidth = []
    zwidth = []

    for file in os.listdir(nifti_dir):
        if file.endswith(".nii"):
            fpath_nifti_dose = os.path.join(nifti_dir, file)
            xwidth_f, ywidth_f, zwidth_f = get_cropped_size(fpath_nifti_dose, dose_level)
            xwidth.append(xwidth_f)
            ywidth.append(ywidth_f)
            zwidth.append(zwidth_f)

    #Display histograms: distribution of bounding box sizes in each dimension
    disp_histogram(xwidth)
    disp_histogram(ywidth)
    disp_histogram(zwidth)

    #Find max width in each dimension
    max_x = max(xwidth)
    max_y = max(ywidth)
    max_z = max(zwidth)

    bounding_box_dims = np.array([max_x, max_y, max_z])
    return max_x, max_y, max_z, bounding_box_dims

'''Crop 3D nifti to user-specified bounding box dimensions. Save as separate nifti file.
Inputs:
    1. fpath_nifti_im: str - full file path to source nifti file that you wish to crop
    2. fpath_nifti_dose: str - full file path to dose cube corresponding to input im containing isodose values to base cropping on
    3. dose_level: float - used to calculate centroid position
    4. bounding_box_dims: either an np array with the width of each dimension (units=pixels)
        OR NaN if you wish to use the isodose bounding box specific to this particular image (i.e. val and test sets)
    5. save_path: str - full file path where cropped nifti should be saved.'''

def crop_nifti(fpath_nifti_im, fpath_nifti_dose, dose_level, bounding_box_dims, save_path):

    sitk_dose = sitk.ReadImage(fpath_nifti_dose)
    sitk_im = sitk.ReadImage(fpath_nifti_im)

    # Create mask image that is just non-background pixels
    fg_mask = (sitk_dose > dose_level)

    #Compute shape statistics on the mask
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(fg_mask)
    c_mm = lsif.GetCentroid(1)
    c_px = fg_mask.TransformPhysicalPointToIndex(c_mm) #centroid in pixels

    if np.isnan(bounding_box_dims[0]):
        bounds = lsif.GetBoundingBox(1)
        # Find max width in each dimension
        xwidth = bounds[3]
        ywidth = bounds[4]
        zwidth = bounds[5]
        bounding_box_dims = np.array([xwidth, ywidth, zwidth])

    #Calculate starting point
    starting_ind = np.floor(c_px - (bounding_box_dims/2))
    starting_ind = starting_ind.astype(int)
    starting_ind = starting_ind.tolist() #Force datatype to int instead of int64

    #Adjust widths if odd and force datatype to int
    for count, i in enumerate(bounding_box_dims):
        if i %2!=0:
            bounding_box_dims[count] = bounding_box_dims[count] + 1
    bounding_box_dims = bounding_box_dims.tolist()
    print(bounding_box_dims)

    extract = sitk.ExtractImageFilter()
    extract.SetSize(bounding_box_dims)
    extract.SetIndex(starting_ind)
    extracted_image = extract.Execute(sitk_im)

    #sanity check
    '''test = sitk.GetArrayFromImage(extracted_image)
    orig = sitk.GetArrayFromImage(sitk_im)
    slicenum = 80
    plt.figure(), plt.imshow(test[slicenum, :, :], cmap='gray'), plt.show()
    plt.figure(), plt.imshow(orig[(slicenum+starting_ind[2]), :, :], cmap="gray"), plt.show()'''

    #Save cropped image as nifti. Overwrite fpath_nifti_im
    sitk.WriteImage(extracted_image, save_path)

#identify source directories/files
split = 'test'
nifti_root = os.path.join('/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_full_TVTsets', split)
dose_dir = os.path.join(nifti_root, 'dose')
im_dir = os.path.join(nifti_root, 'images')
mask_dir = os.path.join(nifti_root, 'rectum_labels_UNET')
dose_level = 18.1

#Identify locations to save cropped files
save_root = os.path.join('/Users/sblackledge/Documents/ProKnow_database/RMH_proknow/proknowPACE/nifti_cropped_TVTsets', split)
dose_dir_save = os.path.join(save_root, 'dose')
im_dir_save = os.path.join(save_root, 'images')
mask_dir_save = os.path.join(save_root, 'rectum_labels')

if split == 'training':
    #Extract max dimensons based on training data
    max_x, max_y, max_z, bounding_box_dims = find_max_dims(dose_dir, dose_level)
else:
    bounding_box_dims = np.empty((3, 1))
    bounding_box_dims[:] = np.NaN

#Loop through all niftis in directory and save cropped versions in a separate directory
for file in os.listdir(dose_dir):
    if file.endswith('.nii'):
        fpath_nifti_dose = os.path.join(dose_dir, file)
        fpath_nifti_im = os.path.join(im_dir, file)
        fpath_nifti_label = os.path.join(mask_dir, file)
        file2 = file+'.gz'
        print(file2)

        #Save cropped images
        '''save_path_im = os.path.join(im_dir_save, file2)
        crop_nifti(fpath_nifti_im, fpath_nifti_dose, dose_level, bounding_box_dims, save_path_im)'''

        #Save cropped labels
        save_path_mask = os.path.join(mask_dir_save, file2)
        crop_nifti(fpath_nifti_label, fpath_nifti_dose, dose_level, bounding_box_dims, save_path_mask)

        #Save cropped dose
        '''save_path_dose = os.path.join(dose_dir_save, file2)
        crop_nifti(fpath_nifti_dose, fpath_nifti_dose, dose_level, bounding_box_dims, save_path_dose)'''
