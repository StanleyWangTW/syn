import glob

import torch
import numpy as np
import nibabel as nib
from nilearn.image import reorder_img, resample_img


# utils
class GetData(torch.utils.data.Dataset):

    def __init__(self, image_dir='', mask_dir='', mode='both'):
        '''mode = image/mask/both
        __getitem__ return {"image": image, "mask": mask}'''

        self.mode = mode

        self.image_ffs = glob.glob(image_dir)
        self.mask_ffs = glob.glob(mask_dir)
        self.mask_ffs.sort()
        self.image_ffs.sort()

    def __len__(self) -> int:
        if self.mode == 'image' or self.mode == 'both':
            return len(self.image_ffs)

        elif self.mode == 'mask':
            return len(self.mask_ffs)

    def __getitem__(self, index) -> dict:
        data = dict()
        if (self.mode == 'image' or self.mode == 'both'):
            data['image'] = nib.load(self.image_ffs[index])

        if (self.mode == 'mask' or self.mode == 'both'):
            data['mask'] = nib.load(self.mask_ffs[index])

        return data


def nib_to_tensor(input_nib, resample='continuous'):
    """
    reorder & resample Nifti to 1mm^3 then convert to 5D torch.Tensor

    return torch.Tensor & nii affine
    """

    input_nib_resp = reorder_img(input_nib, resample=resample)
    input_nib_resp = resample_voxel(input_nib_resp, (1, 1, 1), interpolation=resample)

    vol = torch.from_numpy(input_nib_resp.get_fdata()).float()[None, None, ...]

    return vol, input_nib_resp.affine


def read_nib(input_nib) -> torch.Tensor:
    '''nibabel.NifFI to 5D torch.Tensor'''

    return torch.from_numpy(input_nib.get_fdata()).float()[None, None, ...]


# data preprocessing
def resample_voxel(data_nib, voxelsize, target_shape=None, interpolation='continuous'):
    affine = data_nib.affine
    target_affine = affine.copy()

    factor = np.zeros(3)
    for i in range(3):
        factor[i] = voxelsize[i] / \
            np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i] * factor[i]

    new_nib = resample_img(data_nib,
                           target_affine=target_affine,
                           target_shape=target_shape,
                           interpolation=interpolation)

    return new_nib


def labelize(mask_pred, labels):
    mask_pred_relabel = mask_pred * 0
    for ii in range(len(labels)):
        mask_pred_relabel[mask_pred == ii] = labels[ii]
        # print((ii+1), labels[ii])

    return mask_pred_relabel
