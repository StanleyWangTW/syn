import os
from os.path import join

import numpy as np
import torch
import nibabel as nib
import glob

from training.utils import GetData
from tqdm import tqdm

label_all = dict()
label_all['synthseg'] = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42,
                         43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)
label_all['dgm'] = (0, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54)

def get_dice(mask1, mask2):
    dice = np.sum(mask1 & mask2) * 2
    dice = dice / (1e-6 + np.sum(mask1) + np.sum(mask2))
    return dice

def simon(dir):
    labels = label_all['dgm']
    dataset = GetData(mask_dir=dir, mode='mask')
    output = []

    pbar = tqdm(dataset)
    for data in pbar:
        mask = data['mask'].get_fdata()
        zooms = data['mask'].header.get_zooms()
        vox_sz = zooms[0] * zooms[1] * zooms[2]

        vols = []
        for lb in label_all['synthseg']:
            vols.append((mask == lb).sum().item() * vox_sz)

        # for lb in range(1, 12+1, 2):
        #     vols.append(((mask == lb) | (mask == lb+1)).sum().item() * vox_sz)
            # vols.append(((mask == labels[lb]) | (mask == labels[lb+1])).sum().item() * vox_sz)

        output.append(vols)

    output = np.array(output)
    s, m = output.std(axis=0), output.mean(axis=0)
    # print(s)
    # print(m)

    return (100 * s/m)


def candi123(target_dir, pred_dir):
    dgm_labels = (0, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54)
    dataset = GetData(image_dir=pred_dir, mask_dir=target_dir, mode='both')

    output = []
    for data in tqdm(dataset):
        pred = data['image'].get_fdata()
        target = data['mask'].get_fdata()

        scores = []
        for j in range(1, 12+1, 2):
            # scores.append(get_dice((target == j) | (target == j+1),
            #                        (pred == dgm_labels[j]) | (pred == dgm_labels[j+1])))
            
            scores.append(get_dice((target == j) | (target == j+1),
                                   (pred == j) | (pred == j+1)))
        
    output.append(scores)
    output = np.array(output)
    print(output.mean())
    return output.mean(axis=0)


def downsample(target_dir, pred_dir):
    dgm_labels = (0, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54)

    output = []
    pred_folders = os.listdir(pred_dir)
    for folder in tqdm(pred_folders):
        target_mask = nib.load(join(target_dir, folder)).get_fdata()

        preds = glob.glob(join(pred_dir, folder, '*.nii.gz'))
        for p in preds:
            pred_mask = nib.load(p).get_fdata()

            scores = []
            for j in range(1, 12+1, 2):
                # scores.append(get_dice((target_mask == j)           | (target_mask == j+1),
                #                        (pred_mask == dgm_labels[j]) | (pred_mask == dgm_labels[j+1])))
                
                scores.append(get_dice((target_mask == j) | (target_mask == j+1),
                                       (pred_mask == j)   | (pred_mask == j+1)))
                
            output.append(scores)

    output = np.array(output)
    print(output.shape)
    print(output.mean())
    return output.mean(axis=0)
