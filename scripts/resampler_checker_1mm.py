import os
from os.path import basename, join
import glob
import time
import warnings

import numpy as np
import nibabel as nib
from nilearn.image import reorder_img, resample_img
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


def resample_1mm():
    input_dir = input('input: ')
    input_ffs = glob.glob(input_dir)
    
    interpolation = input('interpolation (image=>"continuous" / mask=>"nearest"): ')
    reorder = input('reorder ? (y/n): ')

    save_dir = input('output folder: ')
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
            if reorder == 'y':
                input_nib = reorder_img(input_nib, resample=interpolation)

            input_nib_resp = resample_voxel(input_nib, (1, 1, 1), interpolation=interpolation)
            print(f'resample {input_nib.header.get_zooms()} to {input_nib_resp.header.get_zooms()}')

            output_file = join(save_dir, basename(f).replace('.nii', '_1mm.nii'))
            nib.save(input_nib_resp, output_file)
            
            s = time.time() - t
            print(f'save to {output_file}, spend {s:.3f} seconds, finish eta. {s * (len(input_ffs) - cnt):.3f} seconds\n')
        except:
            print('! File open error !')

    print('finish')


def check_1mm():
    print('Check 1mm')
    input_dir = input('input: ')
    input_ffs = glob.glob(input_dir)

    print(f'total {len(input_ffs)} files')
    for f in tqdm(input_ffs):
        input_nib = nib.load(f)
        vox_size = input_nib.header.get_zooms()
        if(vox_size != (1, 1, 1)):
            print('\n', vox_size, f)
            return
        
    print('all 1mm')


def main():
    mode = input('Resample or Check ? (r/c)')
    if mode == 'r':
        resample_1mm()
    elif mode == 'c':
        check_1mm()
    else:
        print(f'ERROR: input should be r or c.')


if __name__ == '__main__':
    main()