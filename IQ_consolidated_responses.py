import numpy as np
import pandas as pd
import os
import re


''' Populate 'consolidated_responses' excel spreadsheet with dose statistics from scorecards exported from 
ProKnow database'''

fpath_responses = '/Users/sblackledge/Documents/image_questionnaires/consolidated_responses.xlsx'
new_responses =  '/Users/sblackledge/Documents/image_questionnaires/NEW_consolidated_responses.xlsx'
dir_scorecards = '/Users/sblackledge/Documents/PACE/ScoreCards3'

responses_df = pd.read_excel(fpath_responses)

for file in os.listdir(dir_scorecards):
    if file.endswith(".csv"):
        #Extract relevant information from scorecard
        proknow_number = int(re.findall(r'\d+', file)[0])
        scorecard_df = pd.read_csv(os.path.join(dir_scorecards, file))
        vals = (scorecard_df.Value).values

        #if bowel metrics are nan, assume no bowel contoured because 0cc in high dose region.
        TF = np.isnan(vals[6])
        if TF:
            vals[6:8] = 0

        #Write to 'consolidated responses' file
        responses_df.iloc[proknow_number-2, 11:22] = vals


responses_df.to_excel(new_responses)




