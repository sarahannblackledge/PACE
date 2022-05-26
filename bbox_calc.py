import numpy as np
import SimpleITK as sitk

def bbox_calc(dose_cube_sitk, dose_level):
    #Convert dose cube into mask at specified dose level
    dose_cube = sitk.GetArrayFromImage(dose_cube_sitk)
    mask_cube = dose_cube
    mask_cube[mask_cube < dose_level] = 0
    mask_cube[mask_cube >= dose_level] = 1

    #Write dose mask as nifti for visualization purposes
    '''mask_cube_sitk = sitk.GetImageFromArray(mask_cube)
    mask_cube_sitk.CopyInformation(dose_cube_sitk)

    sitk.WriteImage(mask_cube_sitk, '/Users/sblackledge/Desktop/test_mask.nii')'''

    #Calculate bounding box around dose mask
    r = np.any(mask_cube, axis=(1, 2))
    c = np.any(mask_cube, axis=(0, 2))
    z = np.any(mask_cube, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    dims = [(rmax-rmin), (cmax-cmin), (zmax-zmin)]

    return rmin, rmax, cmin, cmax, zmin, zmax, dims




