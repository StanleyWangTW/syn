import os
from glob import glob
import shutil

path = r'D:\Python_Projects\dataset\SIMON\SIMON_BIDS\sub-032633'
sessions = os.listdir(path)

for ses in sessions:
    ffs = glob(os.path.join(path, ses, 'anat', '*T2star*.nii.gz'))
    print(ses)
    for f in ffs:
        if 'run-1' in f:
            print(os.path.basename(f))
            shutil.copy2(f, r'D:\Python_Projects\dataset\SIMON\T2star')

    print()