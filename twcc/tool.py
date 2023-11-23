import glob

import torch
import numpy as np
import nibabel as nib
from nilearn.image import reorder_img, resample_img
from tqdm import tqdm
import matplotlib.pyplot as plt

from transforms import Rescale


# utils
class GetData(torch.utils.data.Dataset):
    def __init__(self, image_dir='', mask_dir='', mode='both'):
        self.mode = mode
        
        self.image_ffs = glob.glob(image_dir)
        self.mask_ffs = glob.glob(mask_dir)
        self.mask_ffs.sort()
        self.image_ffs.sort()     

    def __len__(self):
        if self.mode == 'image' or self.mode == 'both':
            return len(self.image_ffs)
        
        elif self.mode == 'mask':
            return len(mask_ffs)

    
    def __getitem__(self, index):
        data = dict()
        if(self.mode == 'image' or self.mode == 'both'):
            data['image'] = nib.load(self.image_ffs[index])
            
        if(self.mode == 'mask' or self.mode == 'both'):
            data['mask'] = nib.load(self.mask_ffs[index])
        
        return data
    
def validate(model, dataset, pred_labels, device='cpu'):
    def get_dice(mask1, mask2):
        dice = torch.sum(mask1 & mask2) * 2
        dice = dice / (1e-6 + torch.sum(mask1) + torch.sum(mask2))
        return dice
    
    deepgm_to_aseg = [0, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]
    
    total_scores = 0
    pbar = tqdm(dataset, desc="validating: ")
    with torch.no_grad():
        for data in pbar:
            image = read_nib(data["image"])
            mask = read_nib(data["mask"])
            
            image = image.to(device)
            mask = mask.to(device)[0, 0, ...]
            
            image = Rescale()(image)
            pred_mask = torch.argmax(
                torch.nn.functional.softmax(model(image), dim=1)[0, ...],
                dim=0
            )
        
            pred_scores = 0
            for deepgm in range(1, 12+1):
                pred_scores += get_dice(
                    pred_mask == pred_labels.index(deepgm_to_aseg[deepgm]),
                    mask == deepgm
                ).item()
            
            pbar.set_postfix({'dice': f'{pred_scores / 12: 1.5f}'})
            total_scores += (pred_scores / 12)
            
    return total_scores / len(dataset)


# data preprocessing
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

def nib_to_tensor(input_nib, resample='continuous'):
    input_nib_resp = reorder_img(input_nib, resample=resample)
    input_nib_resp = resample_voxel(input_nib_resp, (1, 1, 1), interpolation=resample)
    
    vol = torch.from_numpy(input_nib_resp.get_fdata()).float()[None, None, ...]
    
    return vol, input_nib_resp.affine

def read_nib(input_nib):
    return torch.from_numpy(input_nib.get_fdata()).float()[None, None, ...]

def labelize(mask_pred, labels):
    mask_pred_relabel = mask_pred * 0
    for ii in range(len(labels)):
        mask_pred_relabel[mask_pred == (ii + 1)] = labels[ii]
        #print((ii+1), labels[ii])
    
    return mask_pred_relabel


# charts plotting
def show_slices(image, layer, cmap, save=False, path=None):
    if type(image) == torch.Tensor:
        image = image.cpu()
        
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.title("Sagittal")
    plt.xlabel(f"Layer: {layer[0]}")
    plt.imshow(np.rot90(image[layer[0], :, :]), cmap=cmap)

    plt.subplot(132)
    plt.title("Coronal")
    plt.xlabel(f"Layer: {layer[1]}")
    plt.imshow(np.rot90(image[:, layer[1], :]), cmap=cmap)

    plt.subplot(133)
    plt.title("Axial")
    plt.xlabel(f"Layer: {layer[2]}")
    plt.imshow(np.rot90(image[:, :, layer[2]]), cmap=cmap)

    plt.tight_layout()

    if save:
        if path is not None:
            plt.savefig(path)
        else:
            print("Path doesn't exist")

    plt.show()

def show_labels(label, layer, nrows, ncols, title, index_all=None, label_dict=None, save=False, path=None):
    """ label: 5D tensor label"""
    plt.figure(figsize=(15, 15))
    for i in range(label.shape[1]):
        plt.subplot(nrows, ncols, i+1)
        if title:
            plt.title(label_dict[index_all[i]])
        plt.imshow(np.rot90(label[0, i, :, :, layer].cpu()), cmap="gray")
    
    plt.tight_layout()
    
    if save:
        if path is not None:
            plt.savefig(path)
        else:
            print("Path doesn't exist")

    plt.show()
