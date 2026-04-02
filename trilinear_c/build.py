from __future__ import print_function
import os
import torch

# try:
#     from torch.utils.ffi import create_extension
#     _ffi_mode = 'legacy'
# except ImportError:
from torch.utils.cpp_extension import load as _torch_load
_ffi_mode = 'cpp'

sources = ['src/trilinear.c']
headers = ['src/trilinear.h']
extra_objects = []
#sources = []
#headers = []
defines = []
with_cuda = False

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/trilinear_cuda.c']
    sources += ['src/trilinear_kernel.cu']
    headers += ['src/trilinear_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

if __name__ == '__main__':
    _torch_load(
        name='_ext.trilinear',
        sources=sources,
        extra_include_paths=[os.path.dirname(os.path.realpath(__file__))],
        define_macros=defines,
        verbose=True,
    )
