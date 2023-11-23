import os

import numpy as np
import nibabel as nib
from nilearn.image import reorder_img
import glob
from tqdm import tqdm

def get_dice(mask1, mask2):
        dice = np.sum(mask1 & mask2) * 2
        dice = dice / (1e-6 + np.sum(mask1) + np.sum(mask2))
        return dice

inputs = glob.glob(os.path.join("..", "dataset", "candi_oasis_aseg", "raw123", "*_dgm.nii.gz"))
scores = [0,] * 12

for i in tqdm(inputs):
    mask = reorder_img(nib.load(i), resample="nearest").get_fdata()
    label = reorder_img(nib.load(i.replace("raw123", "label123").replace("_dgm", "")), resample="nearest").get_fdata()
    for l in range(0, 12):
        scores[l] += get_dice(mask==l+1, label==l+1)

total_score = 0
for i in range(0, 12):
    scores[i] /= len(inputs)
    total_score += scores[i]
    print(f"{i+1}: {scores[i]}")

print(total_score / 12)
