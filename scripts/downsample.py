import os
from os.path import basename, join
import glob
import time
import warnings

import numpy as np
import nibabel as nib
from nilearn.image import reorder_img, resample_img, resample_to_img
from tqdm import tqdm

warnings.filterwarnings("ignore")


def resample_voxel(data_nib, voxelsize, target_shape=None, interpolation='continuous'):
    affine = data_nib.affine
    target_affine = affine.copy()

    factor = np.zeros(3)
    for i in range(3):
        factor[i] = voxelsize[i] / \
            np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i]*factor[i]

    new_nib = resample_img(data_nib, target_affine=target_affine,
                            target_shape=target_shape, interpolation=interpolation)

    return new_nib


def downsample(input_dir, save_dir, res=None, interpolation=None):
    # input_dir = input('input: ')
    input_ffs = glob.glob(input_dir)
    
    # if res is None:
    #     res = int(input('resolution: (3/5/7)'))

    # if interpolation is None:
    #     interpolation = input('interpolation (image=>"continuous" / mask=>"nearest"): ')

    # save_dir = input('output folder: ')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f'total {len(input_ffs)} files')
    cnt = 0
    for f in input_ffs:
        cnt += 1
        t = time.time()
        print(f'{cnt}/{len(input_ffs)} {basename(f)} processing...')
        try:
            input_nib = nib.load(f)
            os.mkdir(os.path.join(save_dir, os.path.basename(f)))

            for i in range(3):
                sample_size = [1, 1, 1]
                sample_size[i] = res

                input_nib_resp = resample_voxel(input_nib, sample_size, interpolation=interpolation)
                print(f'resample {input_nib.header.get_zooms()} to {input_nib_resp.header.get_zooms()}')

                input_nib_resp = resample_to_img(input_nib_resp, input_nib, interpolation=interpolation)
                output_file = join(save_dir, os.path.basename(f), basename(f).replace('.nii', f'_{res}_{i}.nii'))
                nib.save(input_nib_resp, output_file)
            
            s = time.time() - t
            print(f'save to {output_file}, spend {s:.3f} seconds, finish eta. {s * (len(input_ffs) - cnt):.3f} seconds\n')
        except:
            print('! File open error !')

    print('finish')


if __name__ == '__main__':
    input_dir = r'D:\Python_Projects\dataset\candi_oasis_aseg\candi_oasis_aseg_reorder_1mm\raw123\*'
    downsample(input_dir, r'D:\Python_Projects\dataset\candi_oasis_aseg\downsample\3mm', 3, 'continuous')
    downsample(input_dir, r'D:\Python_Projects\dataset\candi_oasis_aseg\downsample\5mm', 5, 'continuous')
    downsample(input_dir, r'D:\Python_Projects\dataset\candi_oasis_aseg\downsample\7mm', 7, 'continuous')