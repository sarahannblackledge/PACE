import tabula
import os
import shutil
import numpy as np
import csv
import pandas as pd
import openpyxl


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
    #rule: if number of characters in string < 4, then should be appended to previous line
    str_lengths = np.char.str_len(structures)
    inds = np.where(np.logical_and(str_lengths > 0, str_lengths < 4))
    inds = inds[0]
    new_names = np.char.add(structures[inds-1], structures[inds])
    structures[inds-1] = new_names
    structures[inds] = ''

    #Add structure name to every line
    for i, name in enumerate(structures):
        if np.char.str_len(name) < 1:
            structures[i] = structures[i-1]

    data_array[:, 0] = structures

    #Sometimes awkward shift where some data that should be in col 9 is in col 10.
    #Duplicate incorrect col10 into col 9 - data in cols10 onward isnt' useful anyways
    col10 = data_array[:,10]

    for i, item in enumerate(col10):
        if '<' in item or '>' in item:
            data_array[i, 9] = item
            data_array[i, 10] = data_array[i, 11]


    #Save as new xlsx file
    df = pd.DataFrame(data_array)
    fpath = os.path.join(patient_dir, fraction_name + '.xlsx')
    df.to_excel(fpath, index=False)

    #Delete temporary csv file
    os.remove(output_path)

def find_criteria_index(dfn_criteria, desired_str):

    inds = []

    for idx, s in enumerate(dfn_criteria):
        if isinstance(s, str):
            if desired_str in s:
                inds.append(idx)
    inds = np.asarray(inds)

    return inds

def populate_dose_stat_xlsx(patient_dir, fraction):
    #get patient name
    patient_name = os.path.split(patient_dir)[1]

    #Duplicate template file for specified patient
    fpath_template_file = '/Users/sblackledge/Documents/audit_evolutivePOTD/dose_stat_template.xlsx'
    fpath_output_file = os.path.join(patient_dir, patient_name + '_dose_stats.xlsx')
    TF = os.path.isfile(fpath_output_file)
    if not TF:
        shutil.copy(fpath_template_file, fpath_output_file)

    # Open file for writing
    workbook = openpyxl.load_workbook(fpath_output_file)
    #Delete sheet1 (automatically created by openpyxl - annoying and unnecessary)
    orig_sheetnames = workbook.sheetnames
    if 'Sheet1' in orig_sheetnames:
        std = workbook['Sheet1']
        workbook.remove(std)

    sheetnames = ['Fraction 1', 'Fraction 2', 'Fraction 3', 'Fraction 4', 'Fraction 5']
    row_start_inds = [3, 17, 31, 45, 59, 73]


    #Extract data from DVS stats excel file - convert to pandas dataframe
    fpath_dvh = os.path.join(patient_dir, patient_name + "_DVH stats_" + str(fraction) + '.xlsx')
    df = pd.read_excel(fpath_dvh)
    dfn = df.to_numpy()
    dfn_structures = dfn[:,0]
    dfn_criteria = dfn[:, 9]
    dfn_criteria = list(dfn_criteria)

    #Correct structures names so all on one line (some are multiline by default)
    for i, s in enumerate(dfn_structures):
        if "\n" in s:
            temp= s.splitlines()
            oneliner = "".join(temp)
            dfn_structures[i] = oneliner

    structure_basenames = ['CTVpsv_4000', 'PTVpsv_3625', 'Bladder', 'Rectum', 'Bowel']
    suffixes = ['', '_CT', '_MR1', '_MR2', '_MR3', '_MR4']

    for f in range(fraction + 1):
        structures = [structure_basename + suffixes[f] for structure_basename in structure_basenames]
        print(structures)

        #Extract dose stats for online plan
        #CTV
        ind0 = np.where(dfn_structures == structures[0])[0][0]
        stats = []
        ctvstats = float(dfn[ind0, 7])
        stats.append(ctvstats)

        #PTV
        inds_structures = np.where(dfn_structures == structures[1])[0] #rows containing requested structure
        criteria = ['V3625', 'D98']
        col_inds = [7, 8]
        for i, c in enumerate(criteria):
            inds_criteria = find_criteria_index(dfn_criteria, c)
            ind = np.intersect1d(inds_structures, inds_criteria)[0]
            val = float(dfn[ind, col_inds[i]])
            stats.append(val)

        #Bladder
        inds_structures = np.where(dfn_structures == structures[2])[0]
        criteria = ['V3700', 'V1810']
        col_inds = [6, 7]
        for i, c in enumerate(criteria):
            inds_criteria = find_criteria_index(dfn_criteria, c)
            ind = np.intersect1d(inds_structures, inds_criteria)[0]
            val = float(dfn[ind, col_inds[i]])
            stats.append(val)

        #Rectum
        inds_structures = np.where(dfn_structures == structures[3])[0]
        criteria = ['V3600', 'V2900', 'V1810']
        col_inds = [6, 7, 7]
        for i, c in enumerate(criteria):
            inds_criteria = find_criteria_index(dfn_criteria, c)
            ind = np.intersect1d(inds_structures, inds_criteria)[0]
            val = float(dfn[ind, col_inds[i]])
            stats.append(val)

        #Bowel
        inds_structures = np.where(dfn_structures == structures[4])[0]
        criteria = ['V3000', 'V1810']
        col_inds = [6, 6]
        for i, c in enumerate(criteria):
            inds_criteria = find_criteria_index(dfn_criteria, c)
            ind = np.intersect1d(inds_structures, inds_criteria)[0]
            val = float(dfn[ind, col_inds[i]])
            stats.append(val)

        print(stats)

        #Set active worksheet for specified fraction
        active_sheet = workbook[sheetnames[fraction-1]]
        print(active_sheet)
        row_start = row_start_inds[f]
        for ii, j in enumerate(stats):
            val = round(row_start + ii)
            active_sheet.cell(row=val, column=3).value = j
        workbook.save(fpath_output_file)


#Loop through fractions
patient_name = 'PAC2011'
patient_dir = os.path.join('/Users/sblackledge/Documents/audit_evolutivePOTD', patient_name)
for fraction in range(1, 6):
    fraction_name = patient_name +'_DVH stats_' + str(fraction)
    pdf_to_csv(fraction_name, patient_dir)
    populate_dose_stat_xlsx(patient_dir, fraction)


#Specific fraction
patient_name = 'PAC2011'
patient_dir = os.path.join('/Users/sblackledge/Documents/audit_evolutivePOTD', patient_name)
fraction = 1
fraction_name = patient_name +'_DVH stats_' + str(fraction)

pdf_to_csv(fraction_name, patient_dir)

populate_dose_stat_xlsx(patient_dir, fraction)

