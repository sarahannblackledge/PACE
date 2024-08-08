import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def extract_hist_vals(index, df_session, df_verif):

    #Extract series from dataframe
    dose_stat_sess = df_session.iloc[1:, index]
    dose_stat_verif = df_verif.iloc[1:, index]

    #Convert to np array
    dose_sess_np = dose_stat_sess.to_numpy(dtype=float)
    dose_verif_np = dose_stat_verif.to_numpy(dtype=float)
    combined = np.concatenate((dose_sess_np, dose_verif_np))

    #Calculate kde
    kde_sess = stats.gaussian_kde(dose_sess_np)
    kde_verif = stats.gaussian_kde(dose_verif_np)
    xx = np.linspace(np.min(combined), np.max(combined), 500)

    #Calculate first and second quartiles
    q1q2q3_sess = [np.quantile(dose_sess_np, 0.25), np.quantile(dose_sess_np, 0.5), np.quantile(dose_sess_np, 0.75)]
    q1q2q3_verif = [np.quantile(dose_verif_np, 0.25), np.quantile(dose_verif_np, 0.5), np.quantile(dose_verif_np, 0.75)]

    # Define bin edges that cover both datasets
    bins = np.linspace(min(min(dose_stat_sess), min(dose_stat_verif)), max(max(dose_stat_sess), max(dose_stat_verif)), 20)

    return dose_stat_sess, dose_stat_verif, bins, kde_sess, kde_verif, xx, q1q2q3_verif

#Read in data
fpath_xlsx = '/Users/sblackledge/Documents/audit_evolutivePOTD/consolidated_dose_stats.xlsx'
df_session = pd.read_excel(fpath_xlsx, sheet_name='Session')
df_verif = pd.read_excel(fpath_xlsx, sheet_name='Verif')

#Hard-code cut-off thresholds, dose stat name, and column index corresponding to excel file as dictionary values
#Bowel always 0, so eliminate from dictionary to avoid useless subplot
cutoff_dict = {
    1: [[85], 'CTVpsv_V4000', 1, 'Volume (%) of CTV covered by 40Gy'],
    2: [[85], 'PTVpsv_V3625', 2, 'Volume (%) of PTV covered by 36.25Gy'],
    3: [[3371.2], 'PTVpsv_D3625*', 3, 'Dose (Gy) covering 98% of PTV'],
    4: [[10], 'Bladder_V3700*', 4, 'Volume (cc) of bladder covered by 37Gy'],
    5: [[40], 'Bladder_V1810', 5, 'Volume (%) of bladder covered by 18.1Gy'],
    6: [[2], 'Rectum_V3600*', 6, 'Volume (cc) of Rectum covered by 36Gy'],
    7: [[20], 'Rectum_V2900', 7, 'Volume (%) of Rectum covered by 29Gy'],
    8: [[50], 'Rectum_V1810', 8, 'Volume (%) of Rectum covered by 18.1Gy'],
    9: [[1], 'Bowel_V3000', 9, 'Volume (cc) of bowel covered by 30Gy'],
    10: [[5], 'Bowel_V1810', 10, 'Volume (cc) of bowel covered by 18.1Gy']
}

# Initialize figure with shared y-axis
fig, axes = plt.subplots(3, 4, figsize=(19, 15), sharey=False)
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Distribution of dose statistics for Target and OAR structures", fontsize=20)

# Initialize lists to collect handles and labels
all_handles = []
all_labels = []

# Loop through dose stats
for n, (ax, ticker) in enumerate(zip(axes.flatten(), cutoff_dict)):
    # Get index from cutoff_dict
    vals = cutoff_dict[n + 1]
    index = vals[2]
    # Get dose stats based on index
    dose_stat_sess, dose_stat_verif, bins, kde_sess, kde_verif, xx, q1q2q3_verif = extract_hist_vals(index, df_session, df_verif)

    # Add histograms
    ax.hist(dose_stat_sess, density=True, bins=bins, alpha=0.5, label='Session', color='gray', histtype='stepfilled')
    ax.hist(dose_stat_verif, density=True, bins=bins, alpha=0.2, label='Verification', color='green', histtype='stepfilled')

    # Add KDE plots
    # ax.plot(xx, kde_sess(xx), color='green')
    # ax.plot(xx, kde_verif(xx), color='purple')

    # Add dashed vertical line at the cut-off value (clinical goals)
    colors = ['black', 'blue', 'brown']
    cutoffs = vals[0]
    for index, c_val in enumerate(cutoffs):
        ax.axvline(c_val, color=colors[index], linestyle='dashed', linewidth=1, label='Clinical goal')

    # Add dashed vertical line at first quartile (targets) and third quartile (OARS) of verification data
    if n <= 2:
        ax.axvline(q1q2q3_verif[0], color='green', linestyle='dashed', linewidth=1)
    else:
        ax.axvline(q1q2q3_verif[2], color='green', linestyle='dashed', linewidth=1, label='Verif 25% (targets) or 75% (OARs)')

    # Set title and x-axis labels for each subplot
    ax.set_title(vals[1])
    ax.set_xlabel(vals[3])

    # Collect handles and labels
    handles, labels = ax.get_legend_handles_labels()
    all_handles.extend(handles)
    all_labels.extend(labels)

# Hide unwanted axes (last two axes)
for ax in axes.flatten()[10:]:
    fig.delaxes(ax)

# Remove duplicates in handles and labels
unique_labels_handles = dict(zip(all_labels, all_handles))
unique_handles = list(unique_labels_handles.values())
unique_labels = list(unique_labels_handles.keys())

# Add a legend in the empty space where the last two axes were
legend_ax = fig.add_subplot(3, 4, 11)
legend_ax.axis('off')  # Hide the axis
legend_ax.legend(unique_handles, unique_labels, loc='upper left', frameon=False, fontsize=18)

# Set common y-axis label
fig.text(0.04, 0.5, 'Probability Density', va='center', rotation='vertical', fontsize=18)

# Show the plot
plt.show()
