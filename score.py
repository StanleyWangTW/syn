from inference.benchmark import simon, candi123, downsample

# output = candi123(r'D:\Python_Projects\dataset\candi_oasis_aseg\candi_oasis_aseg_reorder_1mm\label123\*', \
#                   r'D:\Python_Projects\dataset\tigerbx\dgm\1mm\*')

# print(output)
# print(output.mean())

# s = simon(r'D:\Python_Projects\dataset\tigerbx\aseg\SIMON\PD\*')
# s = simon(r'D:\Python_Projects\dataset\synthseg\SIMON\PD\*')
# print(s.shape)
# print(s, s.mean())

# from glob import glob
# import os

# ffs = glob('*')
# for f in ffs:
#     os.rename(f, f.replace('_synthseg', ''))

output = downsample(
    r'D:\Python_Projects\dataset\candi_oasis_aseg\candi_oasis_aseg_reorder_1mm\label123',
    r'D:\Python_Projects\dataset\tigerbx\dgm\3mm'
)
print(output)

output = downsample(
    r'D:\Python_Projects\dataset\candi_oasis_aseg\candi_oasis_aseg_reorder_1mm\label123',
    r'D:\Python_Projects\dataset\tigerbx\dgm\5mm'
)
print(output)

output = downsample(
    r'D:\Python_Projects\dataset\candi_oasis_aseg\candi_oasis_aseg_reorder_1mm\label123',
    r'D:\Python_Projects\dataset\tigerbx\dgm\7mm'
)
print(output)