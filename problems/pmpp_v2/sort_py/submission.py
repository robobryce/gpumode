"""sort_v2 submission — GPU MODE leaderboard `sort_v2` (robust baseline).

Problem: sort a 1-D float32 tensor ascending, matching torch.sort.

Strategy: CUB DeviceRadixSort on the float keys directly, full 32-bit radix
range. CUB applies the IEEE-754 float->sortable-unsigned transform internally,
so this is correct for ALL float inputs (including negatives / mixed exponents)
with no key truncation and no data-dependent assumptions. Device temp storage
is sized once for the largest shape and reused across calls.

Correctness note: an earlier graph-captured variant truncated the radix range
to 24 bits and cached a data-dependent "pivot" keyed by the input pointer; that
passed local tests + plain benchmark but FAILED the leaderboard ranked run
(stale pivot under allocator pointer reuse -> reversed output on the 100M
shape). This baseline removes that hazard entirely.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

// Persistent device temp storage, grown on demand and reused across calls.
static void*  g_temp       = nullptr;
static size_t g_temp_bytes = 0;

void sort_keys(torch::Tensor input, torch::Tensor output) {
    const int    n     = static_cast<int>(input.numel());
    const float* d_in  = input.data_ptr<float>();
    float*       d_out = output.data_ptr<float>();

    // Query temp storage requirement for the full 32-bit radix sort.
    size_t need = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, need, d_in, d_out, n, 0, sizeof(float) * 8);
    if (need > g_temp_bytes) {
        if (g_temp) cudaFree(g_temp);
        cudaMalloc(&g_temp, need);
        g_temp_bytes = need;
    }

    // Full-width float radix sort. CUB handles IEEE float ordering internally.
    cub::DeviceRadixSort::SortKeys(
        g_temp, need, d_in, d_out, n, 0, sizeof(float) * 8);
}
"""

_CPP_SRC = "void sort_keys(torch::Tensor input, torch::Tensor output);"

_mod = load_inline(
    name="sort_v2_cub_radix",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["sort_keys"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    inp, output = data
    _mod.sort_keys(inp, output)
    return output
