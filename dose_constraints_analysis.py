import pandas as pd
import numpy as np
import os

# Extract dose metric and corresponding measured value from ProKnow score cards
def read_scorecard(fpath):
    df = pd.read_csv(fpath)
    metric = df.iloc[:, 7]
    metric = metric.to_numpy()
    measured_value = df.iloc[:, 10]
    measured_value = measured_value.to_numpy()
    return metric, measured_value

# Test read_scorecard function for example scorecard
fpath = "/Users/sblackledge/Documents/PACE/ScoreCards2/im2.csv"
metric, measured_value = read_scorecard(fpath)
OAR_vals = measured_value[0:7]
target_vals = measured_value[7:11]
print ("OAR_vals: ", OAR_vals)
print ("Target_vals: ", target_vals)
print("metric vals: ", metric)

