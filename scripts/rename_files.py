import glob
import os

r = '_DKT31_CMA_labels'
ffs = glob.glob('*.nii')
for f in ffs:
    if(r in f):
        os.rename(f, f.replace(r, ''))