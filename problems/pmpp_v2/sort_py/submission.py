"""
CUB DeviceRadixSort::SortKeys via precompiled C-ABI shared library.
The .so is compiled directly with nvcc (zero PyTorch build overhead)
and loaded via ctypes. Library load and init deferred to first call
to avoid CUDA init conflicts with multiprocessing spawn context.
"""
import os
import ctypes
import torch
from task import input_t, output_t

_problem_dir = os.path.dirname(os.path.abspath(__file__))
_lib = None


def _ensure_init():
    global _lib
    if _lib is not None:
        return
    _lib = ctypes.CDLL(os.path.join(_problem_dir, 'sort_cuda_ext.so'))
    _lib.sort_cuda_init.argtypes = []
    _lib.sort_cuda_init.restype = ctypes.c_int
    _lib.sort_cuda_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ]
    _lib.sort_cuda_run.restype = ctypes.c_int
    _lib.sort_cuda_init()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32.
    Precompiled .so loaded via ctypes -- zero import-time overhead.
    """
    input_tensor, output_tensor = data
    _ensure_init()
    inp = input_tensor.contiguous()
    _lib.sort_cuda_run(
        output_tensor.data_ptr(),
        inp.data_ptr(),
        inp.numel(),
        ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream)),
    )
    return output_tensor