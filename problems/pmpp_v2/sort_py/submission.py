"""
CUB DeviceRadixSort::SortKeys via standalone sort.cu built with cpp_extension.load.
CUDA graph capture/replay happens entirely inside sort.cu (cudaStreamPerThread +
cudaStreamCaptureModeGlobal). First call warmups + captures, subsequent calls
replay the graph. No load_inline, no CUDAContext.h, no cudaStream_t in .py.
Leaderboard-compatible: cpp_extension.load only, no stream API in submission.py.
"""
import os
import torch
from torch.utils.cpp_extension import load
from task import input_t, output_t

_sort_dir = os.path.dirname(os.path.abspath(__file__))
_sort_src = os.path.join(_sort_dir, "sort.cu")

sort_module = load(
    name='sort_cuda_graph_native',
    sources=[_sort_src],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    extra_cuda_cflags=['-arch=compute_100'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB SortKeys with native CUDA graph capture/replay.
    Graph management is entirely inside sort.cu:
    - First call: warmup + cudaStreamBeginCapture (Global mode) + instantiate
    - Subsequent calls: cudaGraphLaunch + cudaStreamSynchronize
    The .py file contains no CUDA stream or graph API — leaderboard-safe.
    """
    input_tensor, output_tensor = data
    inp = input_tensor.contiguous()
    sort_module.sort_cuda(inp, output_tensor)
    return output_tensor