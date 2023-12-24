import warnings

import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import nibabel as nib

from training import models, utils, plotting, transforms

warnings.filterwarnings("ignore")

checkpoint_f = r'save\checkpoint.pth.tar'
onnx_name = 'synthseg_unet_24.onnx'

pred_labels = labels = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49,
    50, 51, 52, 53, 54, 58, 60
]
print(len(pred_labels))

# device = "cpu"
# print("device:", device)

# torch_model = models.Unet3D(1, len(pred_labels), 24).to(device)
# checkpoint = torch.load(checkpoint_f)
# torch_model.load_state_dict(checkpoint['model_state_dict'])

# torch.onnx.export(
#     torch_model,  # model being run
#     torch.randn(1, 1, 28, 28, 28).to(device),  # model input (or a tuple for multiple inputs)
#     onnx_name,  # where to save the model (can be a file or file-like object)
#     input_names=['input'],  # the model's input names
#     output_names=['output'],  # the model's output names
#     dynamic_axes={
#         'input': [2, 3, 4],
#         'out': [2, 3, 4]
#     })

onnx_model = onnx.load(onnx_name)
onnx.checker.check_model(onnx_model)

test_data = r"test_data\sub-032633_ses-030_acq-32channel_run-2_T2w_1mm.nii.gz"
img = utils.read_nib(nib.load(test_data))
plotting.show_slices(img[0, 0, ...], (100, 100, 100), 'gray')

x = transforms.Rescale()(img).numpy()
ort_sess = ort.InferenceSession(onnx_name, providers=["CPUExecutionProvider"])
logits = ort_sess.run(None, {'input': x})[0][0, ...]
mask_pred = np.argmax(logits, axis=0)
mask_pred = utils.labelize(mask_pred, labels)
print(np.unique(mask_pred), mask_pred.shape)
plotting.show_slices(mask_pred, (100, 100, 170), plotting.get_cmap())
