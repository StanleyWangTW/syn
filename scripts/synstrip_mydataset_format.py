import os
import shutil

src_dir = r'D:\Python_Projects\dataset\synthstrip_data_v1.4\synthstrip_data_v1.4'
src_flist = os.listdir(src_dir)

keywords = ['ixi', 'fsm', 'asl']

sel_flist = list()
for folder in src_flist:
    ff = folder.split('_')
    if ff[0] in keywords and ff[1] == 't1':
        # print(folder)
        sel_flist.append(folder)

print(len(sel_flist))

des_dir = r'D:\Python_Projects\dataset\synthstrip'
if not os.path.isdir(des_dir):
    os.mkdir(des_dir)

image_dir = os.path.join(des_dir, 'image')
label_dir = os.path.join(des_dir, 'label')
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
if not os.path.isdir(label_dir):
    os.mkdir(label_dir)

for folder in sel_flist:
    image_ff = os.path.join(src_dir, folder, 'image.nii.gz')
    label_ff = os.path.join(src_dir, folder, 'labels.nii.gz')
    shutil.copy(image_ff, os.path.join(image_dir, f'{folder}_image.nii.gz'))
    shutil.copy(label_ff, os.path.join(label_dir, f'{folder}_label.nii.gz'))
