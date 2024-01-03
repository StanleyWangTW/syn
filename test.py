import nibabel as nib
import numpy as np

from training import plotting
from tigersyn import run


def clamp_HU(image, MIN_BOUND=-100, MAX_BOUND=400.0):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def min_max_normalization(data):
    # 計算1%和99%的百分位數
    min_val = np.percentile(data, 1)
    max_val = np.percentile(data, 99)
    print(min_val, max_val)

    # 將數據歸一化到0到1的範圍
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data[normalized_data > 1] = 1.
    normalized_data[normalized_data < 0] = 0.

    return normalized_data


img = nib.load(r'CT.nii.gz')
img_clamp = clamp_HU(img.get_fdata())
# img_clamp = min_max_normalization(img.get_fdata())
print(img_clamp.max(), img_clamp.min())

out = nib.nifti1.Nifti1Image(img_clamp, img.affine, img.header)
nib.nifti1.save(out, 'out.nii.gz')

run('s', r'out.nii.gz')

# img = nib.load(r'test_data\sub-032633_ses-030_acq-32channel_run-2_T2w_1mm.nii.gz').get_fdata()
# plotting.show_slices(image=img)

# img = nib.load(r'out_syn.nii.gz').get_fdata()
# plotting.show_slices(image=img, cmap=plotting.get_cmap())
