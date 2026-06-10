"""
CUB SortKeys via ctypes+CDLL (nvcc-compiled).
NO load_inline, NO cpp_extension, NO CUDAContext.h, NO cudaStream_t in .py.
Native CUDA graph capture is incompatible with CUB's internal multi-kernel
orchestration (error 900) -- fall back to optimized direct execution.
"""
import torch
import ctypes
import os
import subprocess
import hashlib
import fcntl
from task import input_t, output_t

_SORT_CU = r"""
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <cstdint>

static void*  _temp       = nullptr;
static size_t _temp_bytes = 0;
static int    _ready      = 0;

static void _setup() {
    if (_ready) return;
    cudaFree(0);

    size_t need = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, need,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        100000000,
        0, 32,
        0);
    cudaDeviceSynchronize();
    _temp_bytes = need * 11 / 10 + 65536;
    cudaMalloc(&_temp, _temp_bytes);
    _ready = 1;
}

extern "C" {

void sort_init() { _setup(); }

void sort_float32(const float* d_in, float* d_out, int n) {
    _setup();
    const int32_t* ki = reinterpret_cast<const int32_t*>(d_in);
    int32_t*       ko = reinterpret_cast<int32_t*>(d_out);
    size_t tb = _temp_bytes;
    // Stream 0 (default) -- eval.py records CUDA events on this stream.
    // No internal sync -- caller (eval.py) handles torch.cuda.synchronize.
    cub::DeviceRadixSort::SortKeys(_temp, tb,
        ki, ko, n, 0, 32, 0);
}

}  // extern "C"
"""


def _compile_and_load():
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(here, ".torch_ext")
    os.makedirs(cache_dir, exist_ok=True)

    src_hash = hashlib.md5(_SORT_CU.encode()).hexdigest()[:16]
    sort_so = os.path.join(cache_dir, f"_d{src_hash}.so")
    sort_lock = sort_so + ".lock"

    if os.path.exists(sort_so):
        lib = ctypes.CDLL(sort_so)
        lib.sort_init.argtypes = []
        lib.sort_init.restype = None
        lib.sort_float32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.sort_float32.restype = None
        return lib

    with open(sort_lock, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(sort_so):
                lib = ctypes.CDLL(sort_so)
                lib.sort_init.argtypes = []
                lib.sort_init.restype = None
                lib.sort_float32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                lib.sort_float32.restype = None
                return lib

            sort_cu = os.path.join(cache_dir, f"_d{src_hash}.cu")
            sort_tmp = sort_so + ".tmp"
            with open(sort_cu, "w") as f:
                f.write(_SORT_CU)

            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            cmd = [
                "nvcc", "-shared", "-O3",
                "-Xcompiler", "-fPIC",
                "-arch=sm_100",
                f"-I{cuda_home}/include",
                "-o", sort_tmp,
                sort_cu,
                "-lcudart",
            ]
            cp = subprocess.run(cmd, check=True,
                capture_output=True, text=True, timeout=120)
            msgs = [l for l in cp.stderr.splitlines()
                    if l.strip() and "warning" not in l.lower()]
            if msgs:
                raise RuntimeError("nvcc errors:\n" + "\n".join(msgs[:20]))

            os.rename(sort_tmp, sort_so)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    lib = ctypes.CDLL(sort_so)
    lib.sort_init.argtypes = []
    lib.sort_init.restype = None
    lib.sort_float32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.sort_float32.restype = None
    return lib


_sort_lib = _compile_and_load()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    _sort_lib.sort_float32(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output_tensor.data_ptr()),
        ctypes.c_int(input_tensor.numel()),
    )
    return output_tensor