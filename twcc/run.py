# +
import os
import glob
import time
import warnings

import torch
import numpy as np
import nibabel as nib
from nilearn.image import reorder_img, resample_img
import matplotlib.pyplot as plt
from tqdm import tqdm

from tool import show_slices, read_nib, GetData
import transforms
import models
import losses
# -

train_dir = dict()
train_dir['mask'] = r"label_samseg_1mm_reorder/*"
test_dir = dict()
test_dir['image'] = r'candi_oasis_aseg_reorder_1mm/raw123/*'
test_dir['mask'] = r'candi_oasis_aseg_reorder_1mm/label123/*'

label_all = dict()
label_all['synsg'] = (
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28,
    41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
)

label_transforms = transforms.Compose([
    transforms.RandomSkullStrip(),
    transforms.LinearDeform(scales=(0.8, 1.2), degrees=(-20, 20), shears=(-0.015, 0.015), trans=(-30, 30)),
#     transforms.NonlinearDeform(max_std=4),
    transforms.NonlinearDeformTio(),
    transforms.RandomCrop(160)
])

image_transforms = transforms.Compose([
    transforms.GMMSample(mean=(0, 255), std=(0, 35)),
    transforms.RandomBiasField(max_std=0.6),
    transforms.Rescale(),
    transforms.GammaTransform(std=0.4),
    transforms.RandomDownSample(max_slice_space=9, alpha=(0.95, 1.05), r_hr=1)
])

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("device:", device)


# Loss Functions    
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class SoftDiceLossWithLogit(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceLossWithLogit, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        y_logit = torch.nn.functional.softmax(y_pred, dim=1)
        intersection = (y_logit * y_true).sum(dim=(2, 3, 4))
        union = y_logit.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))
        return 1 - torch.mean((2 * intersection + self.smooth) / (union + self.smooth))


train_set = GetData(mask_dir=train_dir['mask'], mode='mask')
valid_set = GetData(test_dir['image'], test_dir['mask'], mode='both')
print(f"Training Dataset has {len(train_set)} Nifti images.")
print(f"Test Dataset has {len(valid_set)} Nifti images.")

pred_labels = label_all['synsg']
learning_rate = 1e-4
model = models.Unet3D(1, len(pred_labels), 24).to(device)
print(f"Model predicts {len(pred_labels)} labels.")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = SoftDiceLossWithLogit()


save_dir = "save"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

bestmodel_path = os.path.join(save_dir, "bestmodel.pth")
checkpoint_path = os.path.join(save_dir, "checkpoint.pth.tar")

if input("Load checkpoint ? [y/n]") == "y":
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    loss_list = checkpoint['loss']
    dice_list = checkpoint['dice']
    best_score = checkpoint['best_score']
    print("load checkpoint succesfully")
else:
    start_step = 0
    dice_list = []
    loss_list = []
    best_score = -1

print("starting step: ", start_step)


def validate(model, dataset):
    def get_dice(mask1, mask2):
        dice = torch.sum(mask1 & mask2) * 2
        dice = dice / (1e-6 + torch.sum(mask1) + torch.sum(mask2))
        return dice
    
    deepgm_to_aseg = [0, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]
    
    total_scores = 0
    pbar = tqdm(dataset, desc="validating: ")
    with torch.no_grad():
        for data in pbar:
            image, image_affine = read_nib(data["image"])
            mask, label_affine = read_nib(data["mask"])
            
            image = image.to(device)
            mask = mask.to(device)[0, 0, ...]
            
            pred_mask = torch.argmax(
                torch.nn.functional.softmax(model(image), dim=1)[0, ...],
                dim=0
            )
        
            pred_scores = 0
            for deepgm in range(1, 12+1):
                pred_scores += get_dice(
                    pred_mask == label_all['synsg'].index(deepgm_to_aseg[deepgm]),
                    mask == deepgm
                ).item()
            
            pbar.set_postfix({'dice': f'{pred_scores / 12: 1.5f}'})
            total_scores += (pred_scores / 12)
            
    return total_scores / len(dataset)


savestep = int(input('savestep: '))
steps =  int(input('number of steps: '))

print(f"Number of Steps: {steps}, Learning Rate: {learning_rate}")
print(f"Optimizer: {optimizer}")
print(f"Loss Function: {loss_fn}")
print("Start Training...\n")
torch.cuda.empty_cache()
idx = 0
model.train()
for i in range(steps//savestep):
    total_loss = 0
    pbar = tqdm(range(savestep), desc="training")
    for j in pbar:
        mask, _ = read_nib(train_set[idx]['mask'])
        
        mask = label_transforms(mask.to(device))
        image = image_transforms(mask)
        
        label = transforms.split_labels(mask, pred_labels)

        pred = model(image)
        loss = loss_fn(pred, label)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        idx += 1
        idx = idx if idx < len(train_set) else 0

    with torch.no_grad():
        total_loss /= savestep
        dice_score = validate(model, valid_set)
        
        step = (i+1) * savestep + start_step
        print(f"[{step}/{steps + start_step}] Dice: {dice_score}, Loss: {total_loss}")
        
        if best_score == -1 or dice_score > best_score:
            best_score = dice_score
            torch.save(model, bestmodel_path)
            print("! save best model !")
        
        loss_list.append(total_loss)
        dice_list.append(dice_score)

        # save checkpoint
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_list,
            'dice': dice_list,
            'best_score': best_score
        }, checkpoint_path)
        print("=> save checkpoint")
