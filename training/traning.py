import torch
from tqdm import tqdm

from transforms import Rescale
from utils import read_nib


def validate(model, dataset, device, label_lsit):

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
            mask = mask.to(device)

            # normalize
            mask = Rescale(mask)[0, 0, ...]

            pred_mask = torch.argmax(torch.nn.functional.softmax(model(image), dim=1)[0, ...],
                                     dim=0)

            pred_scores = 0
            for deepgm in range(1, 12 + 1):
                pred_scores += get_dice(pred_mask == label_lsit.index(deepgm_to_aseg[deepgm]),
                                        mask == deepgm).item()

            avg_score = pred_scores / 12
            pbar.set_postfix({'dice': f'{avg_score: 1.5f}'})
            total_scores += avg_score

    return total_scores / len(dataset)
