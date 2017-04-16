import os
import numpy as np
import pandas as pd

from glob import glob

PREPD_FOLDER = 'prepd_stage1/'

label_df = pd.read_csv('sample_truth.csv')

print(label_df)

os.chdir(PREPD_FOLDER)

g = glob('*.npy')
patients = [patient.replace(".npy", "") for patient in g]

test_ids = [patient for patient in patients if patient not in label_df["id"].values]
print(len(test_ids))

for id in test_ids:
    fn = "{}.npy".format(id)
    os.rename(fn, './test/' + fn)

g = glob('*.npy')
shuf = np.random.permutation(g)

nvalid = int(0.2 * len(g))
for i in range(len(g)):
    if i <= nvalid:
        os.rename(shuf[i], './valid/' + shuf[i])
    else:
        os.rename(shuf[i], './train/' + shuf[i])

print("end")
