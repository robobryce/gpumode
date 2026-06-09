"""
CUB DeviceRadixSort::SortKeys via standalone sort.cu built with cpp_extension.load.
Pre-built named .so eliminates per-process load_inline JIT warmup (5-7s per shape).
Persistent temp storage allocated once at module init.
"""
import torch
from torch.utils.cpp_extension import load
from task import input_t, output_t

sort_module = load(
    name='sort_cuda_ext',
    sources=['sort.cu'],
    extra_cuda_cflags=['-O3', '-DNDEBUG'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32.
    No conversion needed — all data is positive IEEE 754 floats.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor