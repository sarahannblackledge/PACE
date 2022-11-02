import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fpath_xlsx = '/Users/sblackledge/Documents/image_questionnaires/consolidated_responses.xlsx'
df = pd.read_excel(fpath_xlsx)

#Image and dose given
responses = pd.DataFrame.to_numpy(df.iloc[1:-1, 6:11])
mean_responses = np.nanmean(responses, axis=1)

plt.figure()
plt.hist(mean_responses, bins=20)
plt.show()

#Image-only
responses = pd.DataFrame.to_numpy(df.iloc[1:-1, 1:6])
mean_responses = np.nanmean(responses, axis=1)

plt.figure()
plt.hist(mean_responses, bins=20)
plt.show()