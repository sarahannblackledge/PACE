import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


'''Read in scorecard exported from proknow with all dose statistics.
Extract values for bowel to see how often it drops into 18 and 30 Gy isodoses

input:
    1. fpath_csv : str - full filepath to individual score card'''
def read_bowel_scorecard(fpath_csv):
    df = pd.read_csv(fpath_csv)
    b18_1 = df.loc[5, 'Value']
    b30 = df.loc[6 ,'Value']
    return b18_1, b30

#Loop through directory and store bowel stats in np arrays
csv_dir = '/Users/sblackledge/Documents/PACE/ScoreCards'
ccBowel_18 = []
ccBowel_30 = []
fnames = []
for file in os.listdir(csv_dir):
    if file.endswith(".csv"):
        fpath_csv = os.path.join(csv_dir, file)
        b18_1, b30 = read_bowel_scorecard(fpath_csv)
        ccBowel_18.append(b18_1)
        ccBowel_30.append(b30)
        fnames.append(file)

ccBowel_18 = np.asarray(ccBowel_18)
ccBowel_30 = np.asarray(ccBowel_30)

#Convert all NaNs (cases where no bowel contour exists) to zero
ccBowel_18[np.isnan(ccBowel_18)] = 0
ccBowel_30[np.isnan(ccBowel_30)] = 0

#Count number of times cc > 0 in 18.1 and 30 Gy
n_18 = np.count_nonzero(ccBowel_18)
n_30 = np.count_nonzero(ccBowel_30)
print(f'Number of cases where cc of bowel covered by 18.1 Gy > 0: {n_18}')
print(f'Number of cases where cc of bowel covered by 30 Gy > 0: {n_30}')

#Plot histogram
n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
fig, axs = plt.subplots(2, 1)
plt.suptitle('Distribution of bowel dose')
axs[0].hist(ccBowel_18, n, alpha=0.9)
axs[0].set(xlabel='Volume (cc) covered by 18.10 Gy')
axs[0].axvline(5, color='r', linestyle='dashed', linewidth=2)
axs[1].hist(ccBowel_30, n, alpha=0.9)
axs[1].set(xlabel='Volume (cc) covered by 30 Gy')
axs[1].axvline(1, color='r', linestyle='dashed', linewidth=2)

for ax in axs.flat:
    ax.set(ylabel='Frequency')

plt.show()