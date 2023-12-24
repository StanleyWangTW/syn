import nibabel as nib

from training import utils, plotting, transforms

labels = (0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46,
          47, 49, 50, 51, 52, 53, 54, 58, 60)
print(len(labels))

label_ff = r'D:\Python_Projects\dataset\synthstrip_iso_reorder\label\fsm_t1_54jc_label_1mm.nii.gz'
mask = utils.read_nib(nib.load(label_ff))

# label_transforms = transforms.Compose([
#     transforms.RandomSkullStrip(),
#     transforms.LinearDeform(),
#     transforms.NonlinearDeformTio(),
#     # transforms.RandomCrop()
# ])

# image_transforms = transforms.Compose([
#     transforms.GMMSample(),
#     transforms.RandomBiasField(),
#     transforms.Rescale(),
#     transforms.GammaTransform(),
#     transforms.RandomDownSample()
# ])

# mask = label_transforms(mask)
# image = image_transforms(mask)
# label = transforms.split_labels(mask, labels)
# print(label.shape)
# print(label.unique())

# plotting.show_slices(image[0, 0, ...], (100, 100, 150), 'gray')
plotting.show_slices(mask[0, 0, ...], (100, 100, 150), cmap=plotting.get_cmap())
# # plotting.show_labels(label, 150, 6, 6)
