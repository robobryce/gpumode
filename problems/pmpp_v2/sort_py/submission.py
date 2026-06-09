"""CUDA sort using CUB DeviceRadixSort.

Direct DeviceRadixSort::SortKeys call - same radix sort torch.sort uses,
but strips out the value-sorting and post-processing overhead.
"""

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// ----- Launch config --------------------------------------------------------
constexpr int SORT_BLOCK_THREADS = 256;
constexpr int SORT_ITEMS_PER_THREAD = 7;  // chosen for good occupancy

// ----- float <-> sortable uint32 ------------------------------------------

__device__ __forceinline__
uint32_t float_to_sortable(float f) {
    uint32_t v = __float_as_uint(f);
    uint32_t mask = (int32_t(v) >> 31) | 0x80000000;
    return v ^ mask;
}

__device__ __forceinline__
float sortable_to_float(uint32_t v) {
    uint32_t mask = (v & 0x80000000) ? 0x80000000 : 0xFFFFFFFF;
    return __uint_as_float(v ^ mask);
}

// ----- Conversion kernels --------------------------------------------------

__global__ void float_to_uint_kernel(
    const float* __restrict__ input,
    uint32_t* __restrict__ output,
    int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        output[idx] = float_to_sortable(input[idx]);
    }
}

__global__ void uint_to_float_kernel(
    const uint32_t* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        output[idx] = sortable_to_float(input[idx]);
    }
}

// ----- Host entry point ----------------------------------------------------

torch::Tensor custom_kernel_fn(torch::Tensor data, torch::Tensor output) {
    int n = data.numel();
    if (n == 0) { output.copy_(data); return output; }

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(data.device());

    // Temp buffer for float -> uint conversion
    auto uint_data = torch::empty({n}, opts_i32);

    // 1. float -> sortable uint32
    int conv_threads = 256;
    int conv_blocks  = std::min((n + conv_threads - 1) / conv_threads, 65535);
    float_to_uint_kernel<<<conv_blocks, conv_threads>>>(
        reinterpret_cast<const float*>(data.data_ptr<float>()),
        reinterpret_cast<uint32_t*>(uint_data.data_ptr<int>()), n);

    // 2. CUB DeviceRadixSort::SortKeys
    // Determine temp storage size
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes,
        static_cast<const uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr),
        n);
    auto cub_temp = torch::empty(
        {static_cast<int64_t>(temp_bytes)}, opts_u8);

    cub::DeviceRadixSort::SortKeys(
        cub_temp.data_ptr(), temp_bytes,
        reinterpret_cast<const uint32_t*>(uint_data.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(uint_data.data_ptr<int>()),
        n,
        0, 32);  // sort all 32 bits

    // 3. sortable uint32 -> float (write to output)
    uint_to_float_kernel<<<conv_blocks, conv_threads>>>(
        reinterpret_cast<const uint32_t*>(uint_data.data_ptr<int>()),
        output.data_ptr<float>(), n);

    return output;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor custom_kernel_fn(torch::Tensor data, torch::Tensor output);
"""

module = load_inline(
    name="cub_device_radix_sort",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["custom_kernel_fn"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    module.custom_kernel_fn(data, output)
    return output