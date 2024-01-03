import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# charts plotting
def show_slices(image, layer=[], cmap='gray', save=False, path=None):
    '''show Nifti 3D data array. Plot Sagittal Coronal Axial planes'''

    if isinstance(image, torch.Tensor):
        image = image.cpu()

    if len(layer) != 3:
        for i in range(3):
            layer.append(image.shape[i] // 2)

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
    plt.colorbar()
    plt.tight_layout()

    if save:
        if path is not None:
            plt.savefig(path)
        else:
            print("Path doesn't exist")

    plt.show()


def show_labels(label,
                layer,
                nrows,
                ncols,
                title=None,
                index_all=None,
                label_dict=None,
                save=False,
                path=None):
    """ label: 5D tensor label"""

    plt.figure(figsize=(15, 15))
    for i in range(label.shape[1]):
        plt.subplot(nrows, ncols, i + 1)
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


# custom cmap for labels plotting
labels = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49,
    50, 51, 52, 53, 54, 58, 60, 100, 101, 102, 103, 104, 105
]

rgb_colors = [(0, 0, 0), (245, 245, 245), (205, 62, 78), (120, 18, 134), (196, 58, 250),
              (220, 248, 164), (230, 148, 34), (0, 118, 14), (122, 186, 220), (236, 13, 176),
              (12, 48, 255), (204, 182, 142), (42, 204, 164), (119, 159, 176), (220, 216, 20),
              (103, 255, 255), (60, 60, 60), (255, 165, 0), (165, 42, 42), (245, 245, 245),
              (205, 62, 78), (120, 18, 134), (196, 58, 250), (220, 248, 164), (230, 148, 34),
              (0, 118, 14), (122, 186, 220), (236, 13, 176), (13, 48, 255), (220, 216, 20),
              (103, 255, 255), (255, 165, 0), (165, 42, 42), (92, 75, 81), (140, 190, 178),
              (242, 235, 191), (243, 181, 98), (240, 96, 96), (102, 187, 106)]

# Convert RGB colors to the range [0, 1]
colors = np.array(rgb_colors) / 255.0
colors = np.zeros([106, 3])
for idx, l in enumerate(labels):
    colors[l] = rgb_colors[idx]
    colors[l] /= 255.0

# Create a ListedColormap
custom_cmap = ListedColormap(colors)


def get_cmap():
    labels = [
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47,
        49, 50, 51, 52, 53, 54, 58, 60
    ]

    # custom cmap for labels plotting
    rgb_colors = [(0, 0, 0), (245, 245, 245), (205, 62, 78), (120, 18, 134), (196, 58, 250),
                  (220, 248, 164), (230, 148, 34), (0, 118, 14), (122, 186, 220), (236, 13, 176),
                  (12, 48, 255), (204, 182, 142), (42, 204, 164), (119, 159, 176), (220, 216, 20),
                  (103, 255, 255), (60, 60, 60), (255, 165, 0), (165, 42, 42), (245, 245, 245),
                  (205, 62, 78), (120, 18, 134), (196, 58, 250), (220, 248, 164), (230, 148, 34),
                  (0, 118, 14), (122, 186, 220), (236, 13, 176), (13, 48, 255), (220, 216, 20),
                  (103, 255, 255), (255, 165, 0), (165, 42, 42)]

    # Convert RGB colors to the range [0, 1]
    colors = np.array(rgb_colors) / 255.0
    colors = np.zeros([labels[-1] + 1, 3])
    for idx, l in enumerate(labels):
        colors[l] = rgb_colors[idx]
        colors[l] /= 255.0

    # Create a ListedColormap
    return ListedColormap(colors)
