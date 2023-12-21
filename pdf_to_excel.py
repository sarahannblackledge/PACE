import tabula
import os
import shutil
import numpy as np
import csv
import pandas as pd

#Convert pdf to csv
def pdf_to_csv(fraction_name, patient_dir):

    ext1 = '.pdf'
    ext2 = '.csv'

    input_path = os.path.join(patient_dir, fraction_name + ext1)
    output_path = os.path.join(patient_dir, fraction_name + '_temp' + ext2)

    tabula.convert_into(input_path, output_path, output_format='csv', pages='all')

    #Fix output so structure labels all on one line, and every line labelled
    with open(output_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data_array = np.array(data)

    structures = data_array[:,0]

    #Fix hanging characters
    #rule: if number of characters in string < 5, then should be appended to previous line
    str_lengths = np.char.str_len(structures)
    inds = np.where(np.logical_and(str_lengths > 0, str_lengths < 5))
    inds = inds[0]
    new_names = np.char.add(structures[inds-1], structures[inds])
    structures[inds-1] = new_names
    structures[inds] = ''

    #Add structure name to every line
    for i, name in enumerate(structures):
        if np.char.str_len(name) < 1:
            structures[i] = structures[i-1]

    data_array[:, 0] = structures

    #Save as new xlsx file
    df = pd.DataFrame(data_array)
    fpath = os.path.join(patient_dir, fraction_name + '.xlsx')
    df.to_excel(fpath, index=False)

    #Delete temporary csv file
    os.remove(output_path)


def populate_dose_stat_xlsx(patient_dir):
    #get patient name
    patient_name = os.path.split(patient_dir)[1]

    #Duplicate template file for specified patient
    fpath_template_file = '/Users/sblackledge/Documents/audit_evolutivePOTD/dose_stat_template.xlsx'
    fpath_output_file = os.path.join(patient_dir, patient_name + '_dose_stats.xlsx')
    shutil.copy(fpath_template_file, fpath_output_file)

    structures = ['CTVpsv_4000', 'PTVpsv_3625', 'Bowel', 'Rectum', 'Bladder']
    fraction = 1
    #online plan




patient_dir = '/Users/sblackledge/Documents/audit_evolutivePOTD/PER042'
fraction_name = 'PER042_DVH stats_3'

pdf_to_csv(fraction_name, patient_dir)