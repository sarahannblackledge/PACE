def get_dicom_tags(dcm, ignore_private=True, ignore_groups=()):
    """ Return a dictionary of all Dicom tags in an input dicom instance.

    Args
    ====
    dcm : pydicom.Dataset.FileDataset
        The loaded dicom file.

    ignore_private : bool (default = True)
        Whether or not to include private tags in the output

    ignore_groups : list (default = ())
        Ignore these dicom groups in the output

    Returns
    =======
    results : dict
        A dictionary containing the tags.
    """
    tags = {}
    for key in list(dcm.keys()):
        g = key.group
        if g % 2 == 1 and ignore_private == True:
            continue # private tag
        if g in ignore_groups:
            continue
        e = key.element
        key_string = "%04x,%04x" % (g, e) # use e.g. int("000a", 16) to convert back to long.
        tags[key_string] = str(dcm[key].value)
    return tags

def copy_dicom_tags(sitk_image, dcm, ignore_private=True, ignore_groups=()):
    tags = get_dicom_tags(dcm, ignore_private=ignore_private, ignore_groups=ignore_groups)
    for key in tags:
        sitk_image.SetMetaData(key, tags[key])