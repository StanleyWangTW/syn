import shutil
import os
from os.path import basename, join
from glob import glob

import nibabel as nib
from tqdm import tqdm

ffs = glob(r'D:\synthseg_pc\dataset\SIMON_data\SIMON_BIDS\sub-032633\ses*\anat\*PD*.nii.gz')
new_dir = r'D:\synthseg_pc\dataset\SIMON_data\PD'

if not os.path.isdir(new_dir):
    os.mkdir(new_dir)

for f in tqdm(ffs):
    shutil.copy(f, join(new_dir, basename(f)))