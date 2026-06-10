"""
CUB SortKeys via ctypes+CDLL (nvcc-compiled). NO graph capture.
NO load_inline, NO cpp_extension, NO CUDAContext.h, NO cudaStream_t in .py.
Default stream (0). Pure ctypes + pre-compiled .so.
"""
import torch
import ctypes
import os
import subprocess
import hashlib
from task import input_t, output_t

# -- embedded CUDA source -----------------------------------------------------
_SORT_CU_SOURCE = r"""
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <algorithm>

extern "C" {

static void  *g_d_temp     = 0;
static size_t g_temp_bytes = 0;

void sort_float32(const float *d_in, float *d_out, int n) {
    size_t cur_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        0, cur_bytes,
        (const int *)0, (int *)0,
        n, 0, 32);

    if (cur_bytes > g_temp_bytes) {
        if (g_d_temp) cudaFree(g_d_temp);
        g_temp_bytes = cur_bytes;
        cudaMalloc(&g_d_temp, g_temp_bytes);
    }

    const int *keys_in  = reinterpret_cast<const int *>(d_in);
    int       *keys_out = reinterpret_cast<int *>(d_out);

    cub::DeviceRadixSort::SortKeys(
        g_d_temp, g_temp_bytes,
        keys_in, keys_out, n,
        0, 32);
}

}  // extern "C"
"""


def _compile_and_load():
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(here, ".torch_ext")
    os.makedirs(cache_dir, exist_ok=True)

    src_hash = hashlib.md5(_SORT_CU_SOURCE.encode()).hexdigest()[:16]
    sort_so = os.path.join(cache_dir, f"_sort_ctypes_{src_hash}.so")

    if os.path.exists(sort_so):
        lib = ctypes.CDLL(sort_so)
        lib.sort_float32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.sort_float32.restype = None
        return lib

    sort_cu = os.path.join(cache_dir, f"_sort_ctypes_{src_hash}.cu")
    with open(sort_cu, "w") as f:
        f.write(_SORT_CU_SOURCE)

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cub_path = os.path.join(cuda_home, "include")

    cmd = [
        "nvcc",
        "-shared", "-O3", "-Xcompiler", "-fPIC",
        "-arch=sm_100",
        f"-I{cub_path}",
        "-o", sort_so,
        sort_cu,
        "-lcudart",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvcc compilation failed:\n{e.stderr}")

    lib = ctypes.CDLL(sort_so)
    lib.sort_float32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.sort_float32.restype = None
    return lib


_sort_lib = _compile_and_load()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    input_tensor = input_tensor.contiguous()
    n = input_tensor.numel()

    _sort_lib.sort_float32(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output_tensor.data_ptr()),
        ctypes.c_int(n),
    )

    return output_tensor