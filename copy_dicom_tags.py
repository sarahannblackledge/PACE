import sys
sys.path.append('/Users/sblackledge/PycharmProjects/PACE/PACE')
from get_python_tags import get_dicom_tags


def copy_dicom_tags(sitk_image, dcm, ignore_private=True, ignore_groups=()):
    tags = get_dicom_tags(dcm, ignore_private=ignore_private, ignore_groups=ignore_groups)
    for key in tags:
        sitk_image.SetMetaData(key, tags[key])